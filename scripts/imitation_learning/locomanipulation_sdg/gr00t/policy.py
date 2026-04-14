# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper


class Policy:
    """Wrapper around GR00T policy for G1 locomanipulation SDG."""

    def __init__(self, model_path: str, embodiment_tag: str):
        """Load the GR00T policy and locomanipulation SDG data config.

        Args:
            model_path: Path to the model checkpoint.
            embodiment_tag: Embodiment tag used by the model (e.g. "new_embodiment").
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tag = EmbodimentTag(embodiment_tag)
        base_policy = Gr00tPolicy(
            embodiment_tag=tag,
            model_path=model_path,
            device=self.device,
        )
        self.policy = Gr00tSimPolicyWrapper(base_policy)
