from collections import OrderedDict
from typing import Dict, cast

import torch
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
    ObservationType,
)
from allenact.base_abstractions.distributions import Distr
from allenact.base_abstractions.misc import ActorCriticOutput


class Imitation(AbstractActorCriticLoss):
    """Expert imitation loss."""

    def __init__(self, uuid: str = "expert_pickupable", action_idx: int = 8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.uuid = uuid
        self.action_idx = action_idx

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[Distr],
        *args,
        **kwargs,
    ):
        """Computes the imitation loss.

        # Parameters

        batch : A batch of data corresponding to the information collected when rolling out (possibly many) agents
            over a fixed number of steps. In particular this batch should have the same format as that returned by
            `RolloutStorage.batched_experience_generator`.
            Here `batch["observations"]` must contain `"expert_action"` observations
            or `"expert_policy"` observations. See `ExpertActionSensor` (or `ExpertPolicySensor`) for an example of
            a sensor producing such observations.
        actor_critic_output : The output of calling an ActorCriticModel on the observations in `batch`.
        args : Extra args. Ignored.
        kwargs : Extra kwargs. Ignored.

        # Returns

        A (0-dimensional) torch.FloatTensor corresponding to the computed loss. `.backward()` will be called on this
        tensor in order to compute a gradient update to the ActorCriticModel's parameters.
        """
        observations = cast(Dict[str, torch.Tensor], batch["observations"])

        losses = OrderedDict()

        should_report_loss = False
        has_observation_to_compute = False

        total_loss = 0
        if self.uuid in observations:
            should_report_loss = True
            has_observation_to_compute = True
            total_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                actor_critic_output.distributions.logits[:, :, self.action_idx],
                observations[self.uuid],
            )

        if not has_observation_to_compute:
            raise NotImplementedError(
                "Imitation loss requires either `expert_action` or `expert_policy`"
                " sensor to be active."
            )
        return (
            total_loss,
            {"expert_cross_entropy": total_loss.item(), **losses} if should_report_loss else {},
        )
