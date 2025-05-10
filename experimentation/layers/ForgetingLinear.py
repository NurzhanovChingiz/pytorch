import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ForgettingLinear(nn.Linear):
    """
    Linear layer that *perturbs* p-fraction of weights on-the-fly:

        • 0.0          - one third of selected indices
        • U(-L, L)     - one third  (Xavier-uniform, L = √(4/(fan_in+fan_out)))
        • 0.7          - one third

    The original weight tensor is **not** modified; masking is applied
    every forward pass while `apply_forgetting` is True.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        p: float = 0.01,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.p = float(p)
        self.apply_forgetting = True

    def forward(self, x: Tensor) -> Tensor:  
        if (not self.apply_forgetting) or (self.p <= 0.0):
            return super().forward(x)

        w = self.weight
        num_w = w.numel()
        num_sel = max(1, int(num_w * self.p))

        device, dtype = w.device, w.dtype

        idx = torch.randperm(num_w, device=device)[:num_sel]
        # Have 3 choice forget, random and overestimate for similar to human memory
        choices = torch.randint(0, 3, (num_sel,), device=device)

        new_vals = torch.zeros(num_sel, dtype=dtype, device=device)
        is_xav   = choices == 1
        if is_xav.any():
            limit = math.sqrt(4.0 / (self.in_features + self.out_features))
            new_vals[is_xav] = torch.empty(
                is_xav.sum(), dtype=dtype, device=device
            ).uniform_(-limit, limit)

        new_vals[choices == 2] = 0.7

        mask = torch.ones_like(w, dtype=dtype)
        mask.view(-1)[idx] = new_vals

        return F.linear(x, w * mask, self.bias)