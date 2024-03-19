import abc
import os
import warnings
from abc import ABC
from collections import defaultdict
from typing import Optional, Sequence, cast, Tuple, Dict, Any, List

import gym
import numpy as np
import prior
import torch
import torch.nn as nn
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.base_abstractions.sensor import ExpertActionSensor, SensorSuite
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import evenly_distribute_count_into_bins
from allenact.utils.system import get_logger
from torch.distributions.utils import lazy_property

from environment.stretch_controller import StretchController
from tasks.task_specs import (
    TaskSpecDatasetList,
    TaskSpecSamplerInfiniteList,
)
from utils.constants.stretch_initialization_utils import (
    STRETCH_ENV_ARGS,
    ALL_STRETCH_ACTIONS,
)
from utils.type_utils import AbstractTaskArgs

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR
from tasks.multi_task_eval_sampler import MultiTaskSampler
from utils.string_utils import get_natural_language_spec


class BaseConfig(ExperimentConfig, ABC):
    MAX_STEPS = 500

    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None

    on_server = torch.cuda.is_available()

    DEFAULT_NUM_TRAIN_PROCESSES: Optional[int] = None
    DEFAULT_TRAIN_GPU_IDS = tuple(range(torch.cuda.device_count()))
    DEFAULT_VALID_GPU_IDS = (torch.cuda.device_count() - 1,)
    DEFAULT_TEST_GPU_IDS = tuple(range(torch.cuda.device_count()))

    NUM_TEST_PROCESSES = 2 * torch.cuda.device_count() if on_server else 1

    TEST_ON_VALIDATION = True

    DEFAULT_LR = 3e-4

    DEFAULT_USE_WEB_RENDER = True

    ACTION_SUPERSET = ALL_STRETCH_ACTIONS

    TRAIN_PROB_RANDOMIZE_MATERIALS = 0.8

    PRELOAD_HOUSES_AND_TASKS = True

    SENSORS = []

    def __init__(
        self,
        num_train_processes: Optional[int] = None,
        num_test_processes: Optional[int] = None,
        test_on_validation: Optional[bool] = None,
        train_gpu_ids: Optional[Sequence[int]] = None,
        val_gpu_ids: Optional[Sequence[int]] = None,
        test_gpu_ids: Optional[Sequence[int]] = None,
        lr: Optional[float] = None,
        pre_download_procthor_houses: bool = True,
        project: Optional[str] = None,
        entity: Optional[str] = None,
    ):
        super().__init__()

        def v_or_default(v, default):
            return v if v is not None else default

        self.num_train_processes = v_or_default(
            num_train_processes, self.DEFAULT_NUM_TRAIN_PROCESSES
        )
        self.num_validation_processes = 1

        self.num_test_processes = v_or_default(num_test_processes, self.NUM_TEST_PROCESSES)

        self.test_on_validation = v_or_default(test_on_validation, self.TEST_ON_VALIDATION)
        self.train_gpu_ids = v_or_default(train_gpu_ids, self.DEFAULT_TRAIN_GPU_IDS)
        self.val_gpu_ids = v_or_default(val_gpu_ids, self.DEFAULT_VALID_GPU_IDS)
        self.test_gpu_ids = v_or_default(test_gpu_ids, self.DEFAULT_TEST_GPU_IDS)

        self.lr = v_or_default(lr, self.DEFAULT_LR)

        self.sampler_devices = self.train_gpu_ids

        self.extra_losses = None

        self.action_space = gym.spaces.Discrete(len(self.ACTION_SUPERSET))

        self.preload_houses_and_tasks = v_or_default(self.PRELOAD_HOUSES_AND_TASKS, True)
        # Note that we don't want to load these information during the online eval
        if self.preload_houses_and_tasks:
            self.split_to_procthor_houses  # Run so that this lazy property is initialized
            self.split_to_tasks  # Run so that this lazy property is initialized

            if pre_download_procthor_houses:
                assert len(self.split_to_procthor_houses["train"]) > 0  # Forces download

        self.project = os.environ.get("WANDB_PROJECT", project)
        self.entity = os.environ.get("WANDB_ENTITY", entity)

    @classmethod
    def tag(cls) -> str:
        return f"ObjectNav-Offline-Base"

    @abc.abstractmethod
    def preprocessors(self):
        raise NotImplementedError

    def create_model(self, **kwargs) -> nn.Module:
        raise NotImplementedError

    @lazy_property
    def split_to_procthor_houses(self):
        if self.on_server:
            max_houses_per_split = {"train": 150000, "val": 15000, "test": 0}
        else:
            max_houses_per_split = {"train": 64, "val": 15000, "test": 0}
        train_houses = prior.load_dataset(
            dataset="spoc-data",
            entity="spoc-robot",
            revision="local-objaverse-procthor-houses",
            path_to_splits=None,
            split_to_path={
                k: os.path.join(OBJAVERSE_HOUSES_DIR, f"{k}.jsonl.gz")
                for k in ["train", "val", "test"]
            },
            max_houses_per_split=max_houses_per_split,
        )

        return train_houses

    @lazy_property
    def split_to_tasks(self):
        raise NotImplementedError

    def machine_params(self, mode="train", **kwargs):
        sampler_devices: Sequence[torch.device] = []
        devices: Sequence[torch.device]

        if mode == "train":
            workers_per_device = 1
            devices = (
                [torch.device("cpu")]
                if not torch.cuda.is_available()
                else cast(Tuple, self.train_gpu_ids) * workers_per_device
            )

            if self.num_train_processes == 0:
                nprocesses = [0] * max(len(devices), 1)
            else:
                nprocesses = evenly_distribute_count_into_bins(
                    self.num_train_processes, max(len(devices), 1)
                )

            sampler_devices = self.sampler_devices

        elif mode == "valid":
            nprocesses = self.num_validation_processes
            devices = [torch.device("cpu")] if not torch.cuda.is_available() else self.val_gpu_ids

        elif mode == "test":
            devices = [torch.device("cpu")] if not torch.cuda.is_available() else self.test_gpu_ids
            # devices = [0]
            nprocesses = evenly_distribute_count_into_bins(
                self.num_test_processes, max(len(devices), 1)
            )

        else:
            raise NotImplementedError("mode must be train, valid or test")

        sensors = [*self.SENSORS]

        if mode != "train":
            sensors = [s for s in sensors if not isinstance(s, ExpertActionSensor)]

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(sensors).observation_spaces,
                preprocessors=self.preprocessors(),
            )
            if len(self.preprocessors()) > 0
            and (
                mode == "train"
                or (
                    (isinstance(nprocesses, int) and nprocesses > 0)
                    or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
                )
            )
            else None
        )

        return MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sampler_devices=(
                sampler_devices if mode == "train" else devices
            ),  # ignored with > 1 gpu_ids
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

    @classmethod
    def make_sampler_fn(self, **kwargs) -> TaskSampler:
        return MultiTaskSampler(**kwargs)

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(np.int32)

    def _get_sampler_args_for_scene_split(
        self,
        split: str,
        houses: List[Dict[str, Any]],
        tasks: List[Dict[str, Any]],
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]],
        seeds: Optional[List[int]],
        deterministic_cudnn: bool,
        include_expert_sensor: bool = True,
    ) -> Dict[str, Any]:
        assert split in ["train", "val", "test"]

        if total_processes > len(houses):
            raise RuntimeError(
                f"Cannot have `total_processes > len(houses)`"
                f" ({total_processes} > {len(houses)}) when `allow_oversample` is `False`."
            )
        elif len(houses) % total_processes != 0:
            if process_ind == 0:  # Only print warning once
                get_logger().warning(
                    f"Number of houses {len(houses)} is not cleanly divisible by the number"
                    f" of processes ({total_processes}). Because of this, not all processes will"
                    f" be fed the same number of houses."
                )

        task_house_id = []
        for task in tasks:
            task_house_id.append(task["house_index"])
            if "natural_language_spec" not in task.keys():
                task["natural_language_spec"] = get_natural_language_spec(task["task_type"], task)

        if self.on_server:
            all_houses = [t["house_index"] for t in tasks]
        else:
            all_houses = [t["house_index"] for t in tasks if t["house_index"] < 64]
        inds = self._partition_inds(len(all_houses), total_processes)
        house_inds = [
            all_houses[idx] for idx in list(range(inds[process_ind], inds[process_ind + 1]))
        ]

        house_inds_set = set(house_inds)

        subhouses = houses.select(house_inds)
        subtasks = [t for t in tasks if t["house_index"] in house_inds_set]

        assert (
            len(subtasks) > 0
        ), f"Process {process_ind} has no tasks, was assigned houses {house_inds[0]} to {house_inds[-1]}."

        if self.on_server:
            device = devices[process_ind % len(devices)]
        else:
            device = None

        task_args = AbstractTaskArgs(
            action_names=self.ACTION_SUPERSET,
            sensors=[
                s
                for s in self.SENSORS
                if (include_expert_sensor or not isinstance(s, ExpertActionSensor))
            ],
            max_steps=self.MAX_STEPS,
        )

        if split == "train":
            house_index_to_task_specs = defaultdict(lambda: [])
            for t in subtasks:
                house_index_to_task_specs[t["house_index"]].append(t)

            task_spec_dataset = TaskSpecSamplerInfiniteList(
                house_index_to_task_specs, shuffle=True, repeat_house_until_forced=True
            )
        else:
            task_spec_dataset = TaskSpecDatasetList(subtasks)

        target_object_types = []
        for t in subtasks:
            if "broad_synset_to_object_ids" in t.keys():
                target_object_types += list(t["broad_synset_to_object_ids"].keys())
        target_object_types = sorted(list(set(target_object_types)))

        return {
            "task_args": task_args,
            "houses": subhouses,
            "house_inds": house_inds,
            "task_spec_sampler": task_spec_dataset,
            "target_object_types": target_object_types,
            "controller_args": STRETCH_ENV_ARGS,
            "controller_type": StretchController,
            "mode": split,
            "device": device,
            "visualize": False,
            "always_allocate_a_new_stretch_controller_when_reset": True,
            "retain_agent_pose": False,
        }

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        split = "train"
        res = self._get_sampler_args_for_scene_split(
            split=split,
            houses=self.split_to_procthor_houses[split],
            tasks=self.split_to_tasks[split],
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["prob_randomize_materials"] = self.TRAIN_PROB_RANDOMIZE_MATERIALS

        return res

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        split = "val"
        res = self._get_sampler_args_for_scene_split(
            split=split,
            houses=self.split_to_procthor_houses[split],
            tasks=self.split_to_tasks[split],
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        return res

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        if self.test_on_validation:
            if not self.test_on_validation:
                warnings.warn(
                    "`test_on_validation` is set to `True` and thus we will run evaluation on the validation set instead."
                    " Be careful as the saved metrics json and tensorboard files **will still be labeled as"
                    " 'test' rather than 'valid'**."
                )
            else:
                warnings.warn(
                    "No test dataset dir detected, running test on validation set instead."
                    " Be careful as the saved metrics json and tensorboard files *will still be labeled as"
                    " 'test' rather than 'valid'**."
                )

            return self.valid_task_sampler_args(
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            )

        else:
            split = "test"
            res = self._get_sampler_args_for_scene_split(
                split=split,
                houses=self.split_to_procthor_houses[split],
                tasks=self.split_to_tasks[split],
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            )
            return res
