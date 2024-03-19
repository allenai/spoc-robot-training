import prior
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.utils.experiment_utils import (
    Builder,
    TrainingPipeline,
    PipelineStage,
    TrainingSettings,
)
from torch import optim
from torch.distributions.utils import lazy_property

from tasks.multi_task_eval_sampler import MultiTaskSampler
from training.online.mixins import SigLIPViTSGRUMixin
from utils.type_utils import RewardConfig


class SigLIPViTSGRUFetchType(SigLIPViTSGRUMixin):
    @classmethod
    def tag(cls):
        return "SigLIP_GRU_FetchType"

    @lazy_property
    def split_to_tasks(self):
        tasks = prior.load_dataset(
            dataset="spoc-data",
            entity="spoc-robot",
            revision="rl-fetchtype-data",
        )
        tasks.val = prior.load_dataset(
            dataset="spoc-data",
            entity="spoc-robot",
            revision="chores-small",
            task_types=["FetchType"],
        ).val
        return tasks

    @classmethod
    def make_sampler_fn(cls, **kwargs):
        kwargs["task_args"]["reward_config"] = RewardConfig(
            step_penalty=0.0,
            goal_success_reward=10.0,
            failed_stop_reward=0.0,
            shaping_weight=1.0,
            reached_horizon_reward=0.0,
            positive_only_reward=False,
        )
        return MultiTaskSampler(**kwargs)

    def training_pipeline(self, **kwargs):
        log_interval_large = self.num_train_processes * 128 * 5 if self.on_server else 1

        batch_steps = int(1e9)

        assert (
            self.ADVANCE_SCENE_ROLLOUT_PERIOD is None
        ), "use STEPS_IN_HOUSE_BEFORE_FORCE_SCENE_ADVANCE instead"

        return TrainingPipeline(
            save_interval=5_000_000,
            metric_accumulate_interval=10_000,
            optimizer_builder=Builder(optim.Adam, dict(lr=self.lr)),
            num_mini_batch=1,
            update_repeats=4,
            max_grad_norm=0.5,
            named_losses={
                "ppo_loss": PPO(**PPOConfig),
            },
            gamma=0.99,
            use_gae=True,
            gae_lambda=0.95,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_loss"],
                    max_stage_steps=batch_steps,
                    training_settings=TrainingSettings(
                        num_steps=128,
                        metric_accumulate_interval=log_interval_large,
                        advance_scene_rollout_period=self.STEPS_IN_HOUSE_BEFORE_FORCE_SCENE_ADVANCE
                        // 128,
                    ),
                ),
            ],
        )
