from typing import Literal, List

import torch

from utils.type_utils import THORActions


def create_causal_mask(T: int, device: torch.device):
    return torch.triu(torch.full([T, T], float("-inf"), device=device), diagonal=1)


def sample_action_index_from_logits(
    logits: torch.Tensor,
    sampling: Literal[
        "greedy", "sample", "sample_done_only_if_argmax", "sample_done_only_if_prob_gt_thresh"
    ],
    action_list: List[str] = None,
) -> torch.Tensor:
    assert len(logits.shape) == 1, f"expected logits to be 1D, got {logits.shape}"
    if sampling == "greedy":
        action_idx = torch.argmax(logits, dim=-1)
    elif sampling == "sample":
        action_idx = torch.distributions.categorical.Categorical(logits=logits).sample()
    elif sampling == "sample_done_only_if_argmax":
        assert action_list is not None, f"action_list must be provided for {sampling}"
        action_idx = torch.distributions.categorical.Categorical(logits=logits).sample()
        # THORActions.done action is really "end"; but checking "done" too if we ever decide to make it "done"
        sampled_done = action_list[action_idx] in [THORActions.done, THORActions.sub_done]
        is_argmax = action_idx == torch.argmax(logits)
        if sampled_done and not is_argmax:
            while action_list[action_idx] in [THORActions.done, THORActions.sub_done]:
                action_idx = torch.distributions.categorical.Categorical(logits=logits).sample()
    elif sampling == "sample_done_only_if_prob_gt_thresh":
        assert action_list is not None, f"action_list must be provided for {sampling}"
        action_idx = torch.distributions.categorical.Categorical(logits=logits).sample()
        sampled_done = action_list[action_idx] in [THORActions.done, THORActions.sub_done]
        probs = torch.softmax(logits, dim=-1)
        is_gt_thresh = probs[action_idx] > 0.3
        if sampled_done and not is_gt_thresh:
            while action_list[action_idx] in [THORActions.done, THORActions.sub_done]:
                action_idx = torch.distributions.categorical.Categorical(logits=logits).sample()
    else:
        raise NotImplementedError(f"unknown sampling method {sampling}")

    return action_idx
