import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union, final, TYPE_CHECKING


if TYPE_CHECKING:
    from environment.stretch_controller import StretchController
    from tasks.abstract_task_sampler import AbstractSPOCTaskSampler

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
#
import gym
import numpy as np

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task

from utils.type_utils import RewardConfig, THORActions
from utils.string_utils import (
    get_natural_language_spec,
    json_templated_task_string,
)
from utils.data_generation_utils.navigation_utils import get_room_id_from_location
from utils.distance_calculation_utils import position_dist
from utils.sel_utils import sel_metric


class AbstractSPOCTask(Task["StretchController"]):
    task_type_str: Optional[str] = None

    def __init__(
        self,
        controller: "StretchController",
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        action_names: List[str],
        reward_config: Optional[RewardConfig] = None,
        house: Optional[Dict[str, Any]] = None,
        collect_observations: bool = True,
        task_sampler: Optional["AbstractSPOCTaskSampler"] = None,
        **kwargs,
    ) -> None:
        self.collect_observations = collect_observations
        self.task_sampler = task_sampler

        super().__init__(
            env=controller,
            sensors=sensors,
            task_info=task_info,
            max_steps=max_steps,
            **kwargs,
        )
        self.controller = controller
        self.house = house
        self.reward_config = reward_config
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.action_names = action_names
        self.last_action_success: Union[bool, int] = -1
        self.last_action_random: Union[bool, int] = -1
        self.last_taken_action_str = ""

        self._metrics = None
        self.observation_history = []

        self.task_info["followed_path"] = [self.controller.get_current_agent_position()]
        self.task_info["agent_poses"] = [self.controller.get_current_agent_full_pose()]
        self.task_info["taken_actions"] = []
        self.task_info["action_successes"] = []

        self.task_info["id"] = (
            self.task_info["task_type"]
            + "_"
            + str(self.task_info["house_index"])
            + "_"
            + str(int(time.time()))
            # + "_"
            # + self.task_info["natural_language_spec"].replace(" ", "")  ths gives error
        )
        if "natural_language_spec" in self.task_info:
            self.task_info["id"] += "_" + self.task_info["natural_language_spec"].replace(" ", "")

        assert (
            task_info["extras"] == {}
        ), "Extra information must exist and is reserved for information collected during task"

        # Set the object filter to be empty, NO OBJECTS RETURN BY DEFAULT.
        # This is all handled intuitively if you use self.controller.get_objects() when you want objects, don't do
        # controller.controller.last_event.metadata["objects"] !
        self.controller.set_object_filter([])

        self.room_poly_map = controller.room_poly_map
        self.room_type_dict = controller.room_type_dict

        self.visited_and_left_rooms = set()
        self.previous_room = None

        self.path: List = []
        self.travelled_distance = 0.0

    def is_successful(self):
        return self.successful_if_done() and self._took_end_action

    @final
    def record_observations(self):
        # This function should be called:
        # 1. Once before any step is taken and
        # 2. Once per step AFTER the step has been taken.
        # This is implemented in the `def step` function of this class

        assert (len(self.observation_history) == 0 and self.num_steps_taken() == 0) or len(
            self.observation_history
        ) == self.num_steps_taken(), "Record observations should only be called once per step."
        self.observation_history.append(self.get_observations())

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self.action_names))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    def close(self) -> None:
        pass

    def step_with_action_str(self, action_name: str, is_random=False):
        assert action_name in self.action_names
        self.last_action_random = is_random
        return self.step(self.action_names.index(action_name))

    def get_observation_history(self):
        return self.observation_history

    def get_current_room(self):
        agent_position = self.controller.get_current_agent_position()
        return get_room_id_from_location(self.room_poly_map, agent_position)

    @final
    def step(self, action: Any) -> RLStepResult:
        if self.num_steps_taken() == 0:
            self.record_observations()
        action_str = self.action_names[action]

        current_room = self.get_current_room()
        if current_room != self.previous_room and current_room is not None:
            if self.previous_room is not None:
                self.visited_and_left_rooms.add(self.previous_room)
            self.previous_room = current_room

        step_result = super().step(action=action)
        self.record_observations()

        position = self.controller.get_current_agent_position()

        self.task_info["taken_actions"].append(action_str)
        self.task_info["followed_path"].append(position)
        self.task_info["agent_poses"].append(self.controller.get_current_agent_full_pose())
        self.task_info["action_successes"].append(self.last_action_success)

        return step_result

    def _step(self, action: int) -> RLStepResult:
        action_str = self.action_names[action]
        self.last_taken_action_str = action_str

        if action_str == THORActions.done:
            self._took_end_action = True
            self._success = self.successful_if_done()
            self.last_action_success = self._success
        elif action_str == THORActions.sub_done:
            self.last_action_success = False
        else:
            event = self.controller.agent_step(action=action_str)
            self.last_action_success = bool(event)

            position = self.controller.get_current_agent_position()
            self.path.append(position)

            if len(self.path) > 1:
                self.travelled_distance += position_dist(
                    p0=self.path[-1], p1=self.path[-2], ignore_y=True
                )
        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def judge(self):
        raise NotImplementedError

    def render(self, mode: Literal["rgb", "depth"] = "rgb", *args, **kwargs) -> np.ndarray:
        raise NotImplementedError(f"Mode '{mode}' is not supported.")

    @abstractmethod
    def successful_if_done(self, strict_success=False) -> bool:
        raise NotImplementedError

    def get_observations(self, **kwargs) -> Any:
        if self.collect_observations:
            obs = super().get_observations()
            return obs
        return None

    def metrics(self) -> Dict[str, Any]:
        # raise NotImplementedError
        if not self.is_done():
            return {}

        metrics = super().metrics()

        metrics["success"] = self._success
        metrics["task_info"] = self.task_info
        metrics["sel"] = (
            sel_metric(
                success=self._success,
                optimal_episode_length=self.task_info["expert_length"],
                actual_episode_length=self.num_steps_taken(),
            )
            if "expert_length" in self.task_info
            else 0
        )
        metrics["sel"] = (
            0.0 if metrics["sel"] is None or np.isnan(metrics["sel"]) else metrics["sel"]
        )

        self._metrics = metrics

        return metrics

    def to_dict(self):
        return self.task_info

    def to_string(self):
        return get_natural_language_spec(self.task_info["task_type"], self.task_info)

    def to_string_templated(self):
        return json_templated_task_string(self.task_info)

    def add_extra_task_information(self, key, value):
        assert (
            key not in self.task_info["extras"]
        ), "Key already exists in task_info['extras'], overwriting is not permitted. Addition only"
        self.task_info["extras"][key] = value
