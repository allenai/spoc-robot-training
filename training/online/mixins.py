import abc
from typing import final, Sequence, Union

import gym
import torch
import torch.nn as nn
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import Builder

from architecture.allenact_preprocessors.siglip_preprocessors import (
    SigLIPPreprocessor,
    DataAugmentationPreprocessor,
)
from architecture.models.allenact_recurrent_models.siglip_gru import SigLIPTensorNavActorCritic
from environment.manipulation_sensors import AnObjectIsInHand
from environment.navigation_sensors import TaskNaturalLanguageSpecSensor
from environment.vision_sensors import (
    RawNavigationStretchRGBSensor,
    RawManipulationStretchRGBSensor,
)
from training.online.base import BaseConfig
from utils.constants.stretch_initialization_utils import INTEL_CAMERA_HEIGHT, INTEL_CAMERA_WIDTH
from utils.wandb_logging import SimpleWandbLogging


class SigLIPViTSGRUMixin(BaseConfig, abc.ABC):
    DEFAULT_NUM_TRAIN_PROCESSES = 64
    DEFAULT_TRAIN_GPU_IDS = tuple(range(torch.cuda.device_count()))

    MAX_STEPS = 600

    SENSORS = [
        RawNavigationStretchRGBSensor(
            width=(INTEL_CAMERA_WIDTH - (INTEL_CAMERA_WIDTH % 32)),
            height=INTEL_CAMERA_HEIGHT,
            uuid="rgb_raw",
        ),
        RawManipulationStretchRGBSensor(
            width=(INTEL_CAMERA_WIDTH - (INTEL_CAMERA_WIDTH % 32)),
            height=INTEL_CAMERA_HEIGHT,
            uuid="manipulation_rgb_raw",
        ),
        TaskNaturalLanguageSpecSensor(uuid="natural_language_spec"),
        AnObjectIsInHand(uuid="an_object_is_in_hand"),
    ]

    STEPS_IN_HOUSE_BEFORE_FORCE_SCENE_ADVANCE = 2000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_validation_processes = 1

    @classmethod
    @final
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = [
            DataAugmentationPreprocessor(
                rgb_input_uuid="rgb_raw",
                output_uuid="rgb",
                normalize=True,
                mean=SigLIPPreprocessor.SIGLIP_RGB_MEANS,
                stdev=SigLIPPreprocessor.SIGLIP_RGB_STDS,
                height=224,
                width=384,
                output_channels=3,
                num_steps_to_change=cls.MAX_STEPS,
            ),
            DataAugmentationPreprocessor(
                rgb_input_uuid="manipulation_rgb_raw",
                output_uuid="manipulation_rgb",
                normalize=True,
                mean=SigLIPPreprocessor.SIGLIP_RGB_MEANS,
                stdev=SigLIPPreprocessor.SIGLIP_RGB_STDS,
                height=224,
                width=384,
                output_channels=3,
                num_steps_to_change=cls.MAX_STEPS,
            ),
            SigLIPPreprocessor(
                rgb_input_uuid="rgb",
                siglip_model_type="ViT-B-16-SigLIP-256",
                output_uuid="rgb_siglip",
                class_emb_only=True,
                input_img_height_width=(256, 256),
                chunk_size=64,
                flatten=False,
                normalize=True,
            ),
            SigLIPPreprocessor(
                rgb_input_uuid="manipulation_rgb",
                siglip_model_type="ViT-B-16-SigLIP-256",
                output_uuid="manipulation_rgb_siglip",
                class_emb_only=True,
                input_img_height_width=(256, 256),
                chunk_size=64,
                flatten=False,
                normalize=True,
            ),
        ]

        return preprocessors

    def create_model(self, **kwargs) -> nn.Module:
        goal_sensor_uuid = next(
            (s.uuid for s in self.SENSORS if isinstance(s, TaskNaturalLanguageSpecSensor)),
            None,
        )

        model = SigLIPTensorNavActorCritic(
            action_space=gym.spaces.Discrete(len(self.ACTION_SUPERSET)),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_siglip_preprocessor_uuid="rgb_siglip",
            manipulation_rgb_siglip_preprocessor_uuid="manipulation_rgb_siglip",
            an_object_is_in_hand_uuid="an_object_is_in_hand",
            arm_proprioception_sensor_uuid=None,
            hidden_size=512,
            goal_dims=32,
            add_prev_actions=True,
            auxiliary_uuids=[],
        )

        return model

    def wandb_logging_callback(self) -> SimpleWandbLogging:
        assert self.entity is not None and self.project is not None, (
            "Entity and project must be set to use wandb logging."
            " Set these values when specifying the --config_kwargs when running the experiment."
        )
        return SimpleWandbLogging(project=self.project, entity=self.entity)
