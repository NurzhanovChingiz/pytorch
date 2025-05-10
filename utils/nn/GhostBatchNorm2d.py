import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostBatchNorm2d(nn.Module):
    """
    Ghost Batch Normalization (Hoffer et al. 2017) for Conv2d outputs.
    Splits the input into virtual micro‑batches, normalises each with its
    own statistics, then stitches the result back together.
    """

    def __init__(
        self,
        num_features: int,
        virtual_batch_size: int = 64,
        momentum: float = 0.01,
        eps: float = 1e-5,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm2d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── fast‑path ────────────────────────────────────────────────────
        if (not self.training) or (x.size(0) <= self.virtual_batch_size):
            return self.bn(x)

        # ── set‑up ───────────────────────────────────────────────────────
        chunks = x.split(self.virtual_batch_size, dim=0)  
        out = torch.empty_like(x)                         
        bn = self.bn                                      
        last_idx = len(chunks) - 1                        
        start = 0

        # ── main loop (single pass, no torch.cat) ───────────────────────
        for idx, c in enumerate(chunks):
            end = start + c.size(0)
            out[start:end] = F.batch_norm(
                c,
                bn.running_mean, bn.running_var,
                bn.weight, bn.bias,
                True,
                0.0 if idx < last_idx else bn.momentum,  
                bn.eps,
            )
            start = end

        return out
