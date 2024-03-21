import copy
import math
import pdb
import random
import warnings
from contextlib import contextmanager
from typing import Dict, Optional, Sequence, List, Tuple, Iterable, Literal

import numpy as np
from ai2thor.controller import Controller
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from shapely import Polygon, GeometryCollection

from environment.spoc_objects import SPOCObject
from utils.constants.stretch_initialization_utils import (
    INTEL_VERTICAL_FOV,
    AGENT_RADIUS_LIST,
    AGENT_MOVEMENT_CONSTANT,
    ADDITIONAL_ARM_ARGS,
    AGENT_ROTATION_DEG,
    WRIST_ROTATION,
    ARM_MOVE_CONSTANT,
    HORIZON,
    ADDITIONAL_NAVIGATION_ARGS,
    STRETCH_COMMIT_ID,
    STRETCH_WRIST_BOUND_1,
    STRETCH_WRIST_BOUND_2,
)
from utils.data_generation_utils.navigation_utils import (
    get_rooms_polymap_and_type,
    get_room_id_from_location,
    rotation_from,
    get_wall_center_floor_level,
    triangulate_room_polygon,
    is_any_object_sufficiently_visible_and_in_center_frame,
    snap_to_skeleton,
)
from utils.distance_calculation_utils import sum_dist_path, position_dist
from utils.synset_utils import is_hypernym_of
from utils.type_utils import THORActions, Vector3


def calc_arm_movement(arm_1, arm_2):
    total_dist = 0
    for k in ["x", "y", "z"]:
        total_dist += (arm_1[k] - arm_2[k]) ** 2

    return total_dist**0.5


class StretchController:
    def __init__(self, initialize_controller=True, **kwargs):
        self.should_render_image_synthesis = (
            kwargs.get("renderDepthImage", False)
            or kwargs.get("renderNormalsImage", False)
            or kwargs.get("renderFlowImage", False)
        )
        self.mode = None

        self.room_poly_map: Optional[Dict[str, Polygon]] = None
        self.room_type_dict: Optional[Dict[str, str]] = None

        if initialize_controller:
            self.controller = Controller(**kwargs)
            self.initialization_args = kwargs
            assert STRETCH_COMMIT_ID in self.controller._build.commit_id

            if "scene" in kwargs:
                self.reset(kwargs["scene"])

            def is_fov_correct():
                return (
                    abs(
                        self.controller.last_event.metadata["thirdPartyCameras"][0]["fieldOfView"]
                        - INTEL_VERTICAL_FOV
                    )
                    < 2
                )

            if not is_fov_correct():
                self.controller.step(
                    "UpdateThirdPartyCamera", thirdPartyCameraId=0, fieldOfView=INTEL_VERTICAL_FOV
                )
                assert is_fov_correct()

    def get_objects_in_hand_sphere(self):
        return self.controller.last_event.metadata["arm"]["pickupableObjects"]

    def get_held_objects(self):
        return self.controller.last_event.metadata["arm"]["heldObjects"]

    def get_arm_sphere_center(self):
        return self.controller.last_event.metadata["arm"]["handSphereCenter"]

    def dist_from_arm_sphere_center_to_obj(self, object_id):
        return position_dist(
            self.get_object_position(object_id), self.get_arm_sphere_center(), ignore_y=False
        )

    def dist_from_arm_sphere_center_to_obj_colliders_closest_to_point(self, object_id):
        arm_sphere_center = self.get_arm_sphere_center()
        points_on_obj = self.controller.step(
            action="PointOnObjectsCollidersClosestToPoint",
            objectId=object_id,
            point=arm_sphere_center,
        ).metadata["actionReturn"]
        if points_on_obj is None or len(points_on_obj) == 0:
            return self.dist_from_arm_sphere_center_to_obj(object_id)
        else:
            dists = [position_dist(arm_sphere_center, p, ignore_y=False) for p in points_on_obj]
        return min(dists)

    @property
    def navigation_camera(self):
        frame = self.controller.last_event.frame
        cutoff = round(frame.shape[1] * 6 / 396)
        return frame[:, cutoff:-cutoff, :]

    @property
    def manipulation_camera(self):
        frame = self.controller.last_event.third_party_camera_frames[0]
        cutoff = round(frame.shape[1] * 6 / 396)
        return frame[:, cutoff:-cutoff, :3]

    @property
    def navigation_camera_segmentation(
        self,
    ):
        if self.controller.last_event.instance_segmentation_frame is None:
            self.controller.step("Pass", renderImageSynthesis=True)
            assert self.controller.last_event.instance_segmentation_frame is not None, (
                "Must pass `renderInstanceSegmentation=True` on initialization"
                " to obtain a navigation_camera_segmentation"
            )

        return self.controller.last_event.instance_masks

    @property
    def manipulation_camera_segmentation(
        self,
    ):
        if self.controller.last_event.instance_segmentation_frame is None:
            self.controller.step("Pass", renderImageSynthesis=True)
            assert self.controller.last_event.instance_segmentation_frame is not None, (
                "Must pass `renderInstanceSegmentation=True` on initialization"
                " to obtain a manipulation_camera_segmentation"
            )

        return self.controller.last_event.third_party_instance_masks[0]

    @property
    def manipulation_depth_frame(self):
        frame = self.controller.last_event.third_party_depth_frames[0]
        cutoff = round(frame.shape[1] * 6 / 396)
        return frame[:, cutoff:-cutoff]

    @property
    def navigation_depth_frame(self):
        frame = self.controller.last_event.depth_frame
        cutoff = round(frame.shape[1] * 6 / 396)
        return frame[:, cutoff:-cutoff]

    def get_segmentation_mask_of_object(
        self, object_id: str, which_camera: Literal["nav", "manip"]
    ):
        if which_camera == "nav":
            segmentation_to_look_at = self.navigation_camera_segmentation
        elif which_camera == "manip":
            segmentation_to_look_at = self.manipulation_camera_segmentation
        else:
            raise NotImplementedError

        if object_id in segmentation_to_look_at:
            mask = segmentation_to_look_at[object_id]
            cutoff = round(mask.shape[1] * 6 / 396)
            result = mask[:, cutoff:-cutoff]
            assert result.shape == self.navigation_camera.shape[:2]
            return result
        else:
            return np.zeros(self.navigation_camera.shape[:2], dtype=bool)

    def get_relative_stretch_current_arm_state(self):
        arm = self.controller.last_event.metadata["arm"]["joints"]
        z = arm[-1]["rootRelativePosition"]["z"]
        x = arm[-1]["rootRelativePosition"]["x"]
        assert abs(x - 0) < 1e-3
        y = arm[0]["rootRelativePosition"]["y"] - 0.16297650337219238
        return dict(x=x, y=y, z=z)

    def step(self, **kwargs):
        if "renderImageSynthesis" not in kwargs:
            kwargs["renderImageSynthesis"] = self.should_render_image_synthesis

        return self.controller.step(**kwargs)

    def get_top_down_path_view(self, agent_path, targets_to_highlight=None):
        if len(self.controller.last_event.third_party_camera_frames) < 2:
            event = self.controller.step({"action": "GetMapViewCameraProperties"})
            cam = event.metadata["actionReturn"].copy()
            cam["orthographicSize"] += 1
            self.controller.step({"action": "AddThirdPartyCamera", "skyboxColor": "white", **cam})

        waypoints = []
        for target in targets_to_highlight or []:
            target_position = self.get_object_position(target)
            target_dict = {
                "position": target_position,
                "color": {"r": 1, "g": 0, "b": 0, "a": 1},
                "radius": 0.5,
                "text": "",
            }
            waypoints.append(target_dict)

        event = self.controller.step(
            {
                "action": "VisualizeWaypoints",
                "waypoints": waypoints,
            }
        )
        # put this over the waypoints just in case
        event = self.controller.step({"action": "VisualizePath", "positions": agent_path})
        self.controller.step({"action": "HideVisualizedPath"})
        path = event.third_party_camera_frames[-1]
        cutoff = round(path.shape[1] * 6 / 396)
        return path[:, cutoff:-cutoff, :]

    def calibrate_agent(self):
        self.step(action="Teleport", horizon=0, standing=True)
        self.step(
            action="RotateCameraMount",
            degrees=27.0 + random.choice(np.arange(-2, 2, 0.2)),
            secondary=False,
        )
        self.step(
            action="ChangeFOV",
            fieldOfView=59 + random.choice(np.arange(-1, 1, 0.1)),
            camera="FirstPersonCharacter",
        )
        self.step(
            action="RotateCameraMount",
            degrees=33.0 + random.choice(np.arange(-2, 2, 0.2)),
            secondary=True,
        )
        self.step(
            action="ChangeFOV",
            fieldOfView=59 + random.choice(np.arange(-1, 1, 0.1)),
            camera="SecondaryCamera",
        )
        self.step(action="SetGripperOpenness", openness=30)

    def reset(self, scene):
        if scene is None:
            raise ValueError("`scene` must be non-None.")

        self.current_scene_json = scene
        self.agent_ids = [i for (i, r) in AGENT_RADIUS_LIST]

        # add metadata here for navmesh?
        base_agent_navmesh = {
            "agentHeight": 1.8,
            "agentSlope": 10,
            "agentClimb": 0.5,
            "voxelSize": 0.1666667,
        }
        scene["metadata"]["navMeshes"] = [
            {**base_agent_navmesh, **{"id": i, "agentRadius": r}} for (i, r) in AGENT_RADIUS_LIST
        ]

        # Mostly for Phone2Proc scenes - may not work but will be corrected if possible in the scene reset.
        if "agent" not in scene["metadata"]:
            scene["metadata"]["agent"] = {
                "horizon": 30,
                "position": {"x": 0, "y": 0.95, "z": 0},
                "rotation": {"x": 0, "y": 270, "z": 0},
                "standing": True,
            }

        scene["metadata"]["agent"]["horizon"] = HORIZON

        reset_event = self.controller.reset(scene=scene)

        self.calibrate_agent()

        # Do not display the unrealistic blue sphere on the agent's gripper
        self.controller.step("ToggleMagnetVisibility", visible=False, raise_for_failure=True)

        self.set_object_filter([])

        self.room_poly_map, self.room_type_dict = get_rooms_polymap_and_type(
            self.current_scene_json
        )

        teleport_event = self.controller.step(
            action="TeleportFull",
            **scene["metadata"]["agent"],  # forceAction=True
        )

        if not teleport_event.metadata["lastActionSuccess"]:
            print("FAILED TO TELEPORT AGENT AFTER INITIALIZATION", scene)
            return teleport_event

        return reset_event

    def get_visible_objects(
        self,
        which_camera: Literal["nav", "manip", "both"] = "nav",
        maximum_distance=2,
        filter_object_ids: Optional[Sequence[str]] = None,
    ):
        assert which_camera in ["nav", "manip", "both"]

        if filter_object_ids is not None and not isinstance(filter_object_ids, List):
            filter_object_ids = list(filter_object_ids)

        visible_objects = set()
        if which_camera in ["nav", "both"]:
            visible_objects.update(
                self.controller.step(
                    "GetVisibleObjects",
                    maxDistance=maximum_distance,
                    objectIds=filter_object_ids,
                ).metadata["actionReturn"]
            )

        if which_camera in ["manip", "both"]:
            visible_objects.update(
                self.controller.step(
                    "GetVisibleObjects",
                    maxDistance=maximum_distance,
                    thirdPartyCameraIndex=0,
                    objectIds=filter_object_ids,
                ).metadata["actionReturn"]
            )

        return list(visible_objects)

    def get_approx_object_mask(
        self, object_id: str, which_camera: Literal["nav", "manip"], divisions: int
    ):
        step_dict = dict(
            action="GetApproxObjectMask",
            objectId=object_id,
            # thirdPartyCameraIndex=None if which_camera == "nav" else 0,
            divisions=divisions,
        )
        if which_camera == "manip":
            step_dict["thirdPartyCameraIndex"] = 0
        return self.step(**step_dict).metadata["actionReturn"]

    def object_is_visible_in_camera(
        self, object_id, which_camera: Literal["nav", "manip", "both"] = "nav", maximum_distance=2
    ):
        return (
            len(
                self.get_visible_objects(
                    which_camera=which_camera,
                    maximum_distance=maximum_distance,
                    filter_object_ids=[object_id],
                )
            )
            != 0
        )

    def get_objects(self) -> List[SPOCObject]:
        with self.include_object_metadata_context():
            return [SPOCObject(o) for o in self.controller.last_event.metadata["objects"]]

    def set_object_filter(self, object_ids: List[str]):
        assert len(object_ids) == 0, "Please don't do this, talk to Luca about why."
        return self.controller.step(
            action="SetObjectFilter",
            objectIds=object_ids,
            raise_for_failure=True,
        )

    def reset_object_filter(self):
        return self.controller.step(action="ResetObjectFilter")

    @contextmanager
    def include_object_metadata_context(self):
        needs_reset = len(self.controller.last_event.metadata["objects"]) == 0
        try:
            if needs_reset:
                self.controller.step("ResetObjectFilter")
                assert self.controller.last_event.metadata["lastActionSuccess"]
            yield None
        finally:
            if needs_reset:
                obj_meta = self.controller.last_event.metadata["objects"]
                self.controller.step("SetObjectFilter", objectIds=[])
                self.controller.last_event.metadata["objects"] = obj_meta
                assert self.controller.last_event.metadata["lastActionSuccess"]

    def get_locations_on_receptacle(self, receptacle_id):
        result = self.step(
            action="GetSpawnCoordinatesAboveReceptacle", objectId=receptacle_id, anywhere=True
        )
        return result.metadata["actionReturn"]

    def get_current_agent_position(self):
        return self.controller.last_event.metadata["agent"]["position"]

    def get_current_agent_full_pose(self):
        return {
            **self.controller.last_event.metadata["agent"],
            "arm": self.controller.last_event.metadata["arm"],
        }

    def get_objects_of_synset_list(
        self,
        target_object_synsets: Iterable[str],
        include_hyponyms: bool,
        all_objs: Optional[List[SPOCObject]] = None,
    ):
        if all_objs is None:
            all_objs = self.get_objects()

        if include_hyponyms:
            return [
                spocobj
                for spocobj in all_objs
                if any(
                    is_hypernym_of(synset=spocobj["synset"], possible_hypernym=other)
                    for other in target_object_synsets
                )
            ]
        else:
            return [spocobj for spocobj in all_objs if spocobj["synset"] in target_object_synsets]

    def get_all_objects_of_synset(
        self, synset: str, include_hyponyms: bool, all_objs: Optional[List[SPOCObject]] = None
    ):
        return self.get_objects_of_synset_list(
            target_object_synsets=[synset], include_hyponyms=include_hyponyms, all_objs=all_objs
        )

    def get_object(self, object_id: str, include_receptacle_info: bool = False):
        """
        NOTE: It may be much less efficient to `include_receptacle_info` than to not.

        :param object_id:
        :param include_receptacle_info:
        :return:
        """
        if include_receptacle_info or any(
            object_id == o["objectId"] for o in self.controller.last_event.metadata["objects"]
        ):
            with self.include_object_metadata_context():
                return SPOCObject(self.controller.last_event.get_object(object_id))

        meta = self.controller.step(
            action="GetObjectMetadata", objectIds=[object_id], raise_for_failure=True
        ).metadata["actionReturn"][0]

        del meta[
            "parentReceptacles"
        ]  # This will always be None when using GetObjectMetadata so remove it so there is no ambiguity
        return SPOCObject(meta)

    def get_obj_pos_from_obj_id(self, object_id):
        return self.get_object(object_id)["axisAlignedBoundingBox"]["center"]

    def get_object_position(self, object_id):
        try:
            return self.get_object(object_id)["position"]
        except KeyError:
            event = self.get_object(object_id)
            print(event)
            print(object_id)

    def get_agent_alignment_to_object(self, object_id: str, use_arm_orientation: bool = False):
        current_agent_pose = self.get_current_agent_full_pose()
        if use_arm_orientation:
            current_agent_pose = copy.deepcopy(current_agent_pose)
            current_agent_pose["rotation"]["y"] += 90
        object_location = self.get_object_position(object_id)
        vector = rotation_from(current_agent_pose, object_location)
        return vector

    def get_agent_alignment_to_wall(self, wall_id, use_arm_orientation: bool = False):
        current_agent_pose = self.get_current_agent_full_pose()
        if use_arm_orientation:
            current_agent_pose = copy.deepcopy(current_agent_pose)
            current_agent_pose["rotation"]["y"] += 90
        wall_location = get_wall_center_floor_level(wall_id, y=current_agent_pose["position"]["y"])
        return rotation_from(current_agent_pose, wall_location)

    def get_reachable_positions(self, grid_size: Optional[float] = None):
        if grid_size is None:
            # Use a smaller grid size than the default as otherwise we may miss many
            # positions that are reachable when not moving with 90 degree rotations
            grid_size = AGENT_MOVEMENT_CONSTANT * 0.75

        rp_event = self.controller.step(action="GetReachablePositions", gridSize=grid_size)
        if not rp_event:
            # NOTE: Skip scenes where GetReachablePositions fails
            warnings.warn(f"GetReachablePositions failed in {self.current_scene_json}")
            return []
        reachable_positions = rp_event.metadata["actionReturn"]
        return reachable_positions

    def stop(self):
        self.controller.stop()

    def agent_step(self, action):
        agents_full_pose_before_action = copy.deepcopy(
            dict(
                agent_pose=self.get_current_agent_full_pose(),
                arm_pose=self.get_relative_stretch_current_arm_state(),
                wrist=self.get_arm_wrist_rotation(),
            )
        )

        if action == THORActions.move_ahead:
            action_dict = dict(action="MoveAgent", ahead=AGENT_MOVEMENT_CONSTANT)
        elif action == THORActions.move_back:
            action_dict = dict(action="MoveAgent", ahead=-AGENT_MOVEMENT_CONSTANT)
        elif action in [
            THORActions.rotate_left,
            THORActions.rotate_right,
            THORActions.rotate_left_small,
            THORActions.rotate_right_small,
        ]:  #  add for smaller rotations
            if action == THORActions.rotate_right:
                degree = AGENT_ROTATION_DEG
            elif action == THORActions.rotate_left:
                degree = -AGENT_ROTATION_DEG
            elif action == THORActions.rotate_right_small:
                degree = AGENT_ROTATION_DEG / 5
            elif action == THORActions.rotate_left_small:
                degree = -AGENT_ROTATION_DEG / 5
            else:
                raise NotImplementedError

            action_dict = dict(action="RotateAgent", degrees=degree)
        elif action in [
            THORActions.move_arm_down,
            THORActions.move_arm_in,
            THORActions.move_arm_out,
            THORActions.move_arm_up,
            THORActions.move_arm_down_small,
            THORActions.move_arm_in_small,
            THORActions.move_arm_out_small,
            THORActions.move_arm_up_small,
        ]:
            base_position = self.get_relative_stretch_current_arm_state()
            change_value = ARM_MOVE_CONSTANT
            small_change_value = ARM_MOVE_CONSTANT / 5
            if action == THORActions.move_arm_up:
                base_position["y"] += change_value
            elif action == THORActions.move_arm_down:
                base_position["y"] -= change_value
            elif action == THORActions.move_arm_out:
                base_position["z"] += change_value
            elif action == THORActions.move_arm_in:
                base_position["z"] -= change_value
            elif action == THORActions.move_arm_up_small:
                base_position["y"] += small_change_value
            elif action == THORActions.move_arm_down_small:
                base_position["y"] -= small_change_value
            elif action == THORActions.move_arm_out_small:
                base_position["z"] += small_change_value
            elif action == THORActions.move_arm_in_small:
                base_position["z"] -= small_change_value
            action_dict = dict(
                action="MoveArm",
                position=dict(x=base_position["x"], y=base_position["y"], z=base_position["z"]),
            )
        elif action in [
            THORActions.wrist_open,
            THORActions.wrist_close,
        ]:
            curr_wrist = self.get_arm_wrist_rotation()
            if action == THORActions.wrist_open:
                rotation_value = -1 * min(
                    WRIST_ROTATION, abs(curr_wrist - (STRETCH_WRIST_BOUND_2 + 360))
                )
            else:  # action == THORActions.wrist_close:
                rotation_value = min(WRIST_ROTATION, abs(STRETCH_WRIST_BOUND_1 - curr_wrist))

            action_dict = dict(action="RotateWristRelative", yaw=rotation_value)
        elif action == THORActions.pickup:
            action_dict = dict(action="PickupObject")
        elif action == THORActions.dropoff:
            action_dict = dict(action="ReleaseObject")
        else:
            print("Action not defined")
            pdb.set_trace()
            raise NotImplementedError("Action not defined")

        if action_dict["action"] in ["RotateWristRelative", "MoveArm"]:
            action_dict = {**action_dict, **ADDITIONAL_ARM_ARGS}
        elif action_dict["action"] == "MoveAgent":
            action_dict = {**action_dict, **ADDITIONAL_NAVIGATION_ARGS}

        event = self.step(**action_dict)

        if action == THORActions.dropoff:
            self.step(action="AdvancePhysicsStep", simSeconds=2)

        agents_full_pose_after_action = copy.deepcopy(
            dict(
                agent_pose=self.get_current_agent_full_pose(),
                arm_pose=self.get_relative_stretch_current_arm_state(),
                wrist=self.get_arm_wrist_rotation(),
            )
        )

        # test for checking move arm is failing or not
        #  return false if arm move  is called but pose is not changed
        if action in THORActions.ARM_ACTIONS:
            event.metadata["lastActionSuccess"] = (
                calc_arm_movement(
                    agents_full_pose_before_action["arm_pose"],
                    agents_full_pose_after_action["arm_pose"],
                )
                > 1e-3
            )

        if action in [
            THORActions.wrist_open,
            THORActions.wrist_close,
        ]:
            event.metadata["lastActionSuccess"] = (
                abs(
                    agents_full_pose_before_action["wrist"] - agents_full_pose_after_action["wrist"]
                )
                > 1e-3
            )

        # Only a failure moving if we don't move enough
        if action in THORActions.MOVE_ACTIONS:
            event.metadata["lastActionSuccess"] = (
                position_dist(
                    agents_full_pose_before_action["agent_pose"]["position"],
                    agents_full_pose_after_action["agent_pose"]["position"],
                )
                > 1e-2
            )

        # Only a failure rotating if we don't rotate enough
        if action in THORActions.ROTATE_ACTIONS:
            event.metadata["lastActionSuccess"] = (
                IThorEnvironment.rotation_dist(
                    agents_full_pose_before_action["agent_pose"]["rotation"],
                    agents_full_pose_after_action["agent_pose"]["rotation"],
                )
                > 2
            )
        return event

    def get_arm_wrist_position(self):
        joint = self.controller.last_event.metadata["arm"]["joints"][-1]
        assert joint["name"] == "stretch_robot_wrist_2_jnt"
        return [joint["rootRelativePosition"][k] for k in ["x", "y", "z"]]

    def get_arm_wrist_absolute_position(self):
        joint = self.controller.last_event.metadata["arm"]["joints"][-1]
        assert joint["name"] == "stretch_robot_wrist_2_jnt"
        return [joint["position"][k] for k in ["x", "y", "z"]]

    def get_arm_wrist_rotation(self):
        joint = self.controller.last_event.metadata["arm"]["joints"][-1]
        assert joint["name"] == "stretch_robot_wrist_2_jnt"
        return math.fmod(
            joint["rootRelativeRotation"]["w"] * joint["rootRelativeRotation"]["y"], 360
        )

    def get_arm_proprioception(self):
        arm_position = self.get_arm_wrist_position()
        arm_rotation = self.get_arm_wrist_rotation()
        full_pose = arm_position + [arm_rotation]
        return full_pose

    # calculate the shortest path to that location
    def get_shortest_path_to_object(
        self,
        object_id,
        initial_position=None,
        initial_rotation=None,
        specific_agent_meshes=None,
        attempt_path_improvement: bool = True,
    ) -> Optional[List[Vector3]]:
        """
        Computes the shortest path to an object from an initial position using a controller

        :param object_id: string with id of the object
        :param initial_position: dict(x=float, y=float, z=float) with the desired initial rotation
        :param initial_rotation: dict(x=float, y=float, z=float) representing rotation around axes or None
        :return:
        """
        if specific_agent_meshes is None:
            specific_agent_meshes = self.agent_ids

        if initial_position is None:
            initial_position = self.get_current_agent_position()

        for nav_mesh_id in specific_agent_meshes:
            args = dict(
                action="GetShortestPath",
                objectId=object_id,
                position=initial_position,
                navMeshId=nav_mesh_id,  # update to incorporate navmesh
            )
            if initial_rotation is not None:
                args["rotation"] = initial_rotation
            event = self.step(**args)
            if event.metadata["lastActionSuccess"]:
                corners = event.metadata["actionReturn"]["corners"]
                if len(corners) == 0:
                    continue
                self.last_successful_path = corners

                if attempt_path_improvement and len(corners) > 2:
                    corners = snap_to_skeleton(
                        controller=self,
                        corners=corners,
                    )

                return corners  # This will slow down data generation

        return None

    def does_some_shortest_path_to_object_exist(
        self,
        object_id: str,
        initial_position=None,
        initial_rotation=None,
    ) -> bool:
        """
        Checks if a shortest path to an object from an initial position exists. This is faster than
        `get_shortest_path_to_object` as we will only use the most general nav mesh.

        :param object_id: string with id of the object
        :param initial_position: dict(x=float, y=float, z=float) with the desired initial rotation
        :param initial_rotation: dict(x=float, y=float, z=float) representing rotation around axes or None
        :return:
        """
        return (
            self.get_shortest_path_to_object(
                object_id=object_id,
                initial_position=initial_position,
                initial_rotation=initial_rotation,
                specific_agent_meshes=[self.agent_ids[-1]],
                attempt_path_improvement=False,
            )
            is not None
        )

    def get_shortest_path_to_point(
        self,
        target_position,
        initial_position=None,
        initial_rotation=None,
        specific_agent_meshes=None,
        attempt_path_improvement=True,
    ):
        """
        Computes the shortest path to an object from an initial position using a controller
        :param target_position: dict(x=float, y=float, z=float) with the desired target position
        :param initial_position: dict(x=float, y=float, z=float) with the desired initial rotation
        :param initial_rotation: dict(x=float, y=float, z=float) representing rotation around axes or None
        :param specific_agent_meshes: list of strings with the specific agent meshes to use: 3 for agent
        circumference, lower for more conservative
        :param attempt_path_improvement: whether to attempt to improve the path by snapping to the skeleton
        """
        if specific_agent_meshes is None:
            specific_agent_meshes = self.agent_ids
        if initial_position is None:
            initial_position = self.get_current_agent_position()

        for nav_mesh_id in specific_agent_meshes:
            args = dict(
                action="GetShortestPathToPoint",
                position=initial_position,
                target=target_position,
                navMeshId=nav_mesh_id,  # update to incorporate navmesh
            )
            if initial_rotation is not None:
                args["rotation"] = initial_rotation
            event = self.step(**args)
            if event.metadata["lastActionSuccess"]:
                corners = event.metadata["actionReturn"]["corners"]
                if len(corners) == 0:
                    continue
                self.last_successful_path = corners

                if attempt_path_improvement and len(corners) > 2:
                    corners = snap_to_skeleton(
                        controller=self,
                        corners=corners,
                    )

                return corners  # This will slow down data generation

        return None

    def is_object_visible_enough_for_interaction(self, object_id: str, manipulation_camera=True):
        return is_any_object_sufficiently_visible_and_in_center_frame(
            controller=self,
            object_ids=[object_id],
            manipulation_camera=manipulation_camera,
        )

    def get_closest_object_from_ids(self, object_ids, return_id_and_dist: bool = False):
        all_paths = [
            (
                obj_id,
                self.get_shortest_path_to_object(
                    obj_id,
                    specific_agent_meshes=[self.agent_ids[-1]],
                    attempt_path_improvement=False,
                ),
            )
            for obj_id in object_ids
        ]

        min_dist = float("inf")
        closest_obj_id = None
        for obj_id, path in all_paths:
            if path is None:
                continue
            dist = sum_dist_path(path)
            if dist < min_dist:
                min_dist = dist
                closest_obj_id = obj_id
        return closest_obj_id if not return_id_and_dist else (closest_obj_id, min_dist)

    def get_candidate_points_in_room(
        self,
        room_id,
        room_triangles: Optional[GeometryCollection] = None,
    ):
        polygon = self.room_poly_map[room_id]

        if room_triangles is None:
            # Triangulates the room, and takes the centers of all triangles as possible
            # target locations
            room_triangles = triangulate_room_polygon(polygon)

        candidate_points = [
            ((t.centroid.x, t.centroid.y), t.area) for t in room_triangles  # type:ignore
        ]

        # We sort the triangles by size so we try to go to the center of the largest triangle first
        candidate_points.sort(key=lambda x: x[1], reverse=True)
        candidate_points = [p[0] for p in candidate_points]

        # The centroid of the whole room polygon need not be in the room when the room is concave. If it is,
        # let's make it the first point we try to navigate to.
        if polygon.contains(polygon.centroid):
            candidate_points.insert(0, (polygon.centroid.x, polygon.centroid.y))

        return candidate_points

    def get_shortest_path_to_room(
        self,
        room_id,
        specific_agent_meshes=None,
        max_tries: int = 5,
        room_triangles: Optional[GeometryCollection] = None,
    ):
        assert max_tries > 0

        candidate_points = self.get_candidate_points_in_room(
            room_id=room_id,
            room_triangles=room_triangles,
        )

        current_agent_position = self.controller.last_event.metadata["agent"]["position"]
        y = current_agent_position["y"]

        if specific_agent_meshes is None:
            specific_agent_meshes = self.agent_ids
        specific_agent_meshes = sorted(specific_agent_meshes)

        path = None
        for agent_id in specific_agent_meshes:
            for point in candidate_points[:max_tries]:
                path = self.get_shortest_path_to_point(
                    target_position=dict(x=point[0], y=y, z=point[1]),
                    initial_position=current_agent_position,
                    specific_agent_meshes=[agent_id],
                    attempt_path_improvement=False,
                )
                if path is not None:
                    break
        return path

    def get_objects_room_id_and_type(self, object_id: str) -> Tuple[str, str]:
        object_position = self.get_object_position(object_id)
        room_id = get_room_id_from_location(self.room_poly_map, object_position)
        room_type_return = (
            self.room_type_dict[room_id] if room_id is not None else None
        )  # making it more robust to none style cases
        return room_id, room_type_return

    def find_closest_room_of_list(self, room_ids, return_id_and_dist: bool = False):
        all_paths = []
        for room_id in room_ids:
            path = self.get_shortest_path_to_room(
                room_id, specific_agent_meshes=[self.agent_ids[-1]]
            )
            all_paths.append((room_id, path))

        min_dist = float("inf")
        closest_room_id = None
        for room_id, path in all_paths:
            if path is None:
                continue
            dist = sum_dist_path(path)
            if dist < min_dist:
                min_dist = dist
                closest_room_id = room_id

        return closest_room_id if not return_id_and_dist else (closest_room_id, min_dist)


class StretchStochasticController(StretchController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_rand_action_kwargs = None

    def step(self, **kwargs):
        if "action" in kwargs and kwargs["action"] in ["MoveAhead", "RotateAgent"]:
            rand = np.random.normal(0, 1, 1)[0]

            if "action" in kwargs and kwargs["action"] == "MoveAgent":
                kwargs["ahead"] += 0.01 * rand
            if "action" in kwargs and kwargs["action"] == "RotateAgent":
                kwargs["degrees"] += 0.5 * rand

            self.last_rand_action_kwargs = kwargs
        return super(StretchStochasticController, self).step(**kwargs)
