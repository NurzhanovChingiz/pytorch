import torch
import torch.nn as nn
import torch.nn.functional as F

def tempered_softmax(logits: torch.Tensor, t: float, dim: int = -1, eps: float = 1e-7):
    if abs(t - 1.0) < 1e-7:
        return torch.softmax(logits, dim=dim)
    #   exp_t(x) = [1 + (1-t)x]_(+)^{1/(1-t)}
    # clip for numerical stability
    logits = logits - logits.max(dim=dim, keepdim=True).values      
    exp_t = torch.relu(1 + (1 - t) * logits).pow(1 / (1 - t))
    return exp_t / (exp_t.sum(dim=dim, keepdim=True) + eps)

class FocalCosineBTLoss(nn.Module):
    """
    Focal‑Cosine loss with **Bi‑Tempered cross‑entropy** (Papyan+ 2019)
    and optional label smoothing.
    """

    def __init__(self,
                 alpha: float = 1.0,
                 gamma: float = 2.0,
                 xent:  float = 0.1,
                 reduction: str = "mean",
                 t1: float = 0.9,
                 t2: float = 1.5,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.alpha, self.gamma, self.xent = alpha, gamma, xent
        self.t1, self.t2 = t1, t2
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    # ---------- Bi‑Tempered cross‑entropy ----------------------------
    def _bt_ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        y = F.one_hot(target, num_classes).float()

        if self.label_smoothing > 0:
            smooth = self.label_smoothing / num_classes
            y = y * (1.0 - self.label_smoothing) + smooth

        p_t2 = tempered_softmax(logits, t=self.t2, dim=-1)
        log_p_t1 = (p_t2.pow(1 - self.t1) - 1) / (1 - self.t1)          # log_t1

        loss = -torch.sum(y * log_p_t1, dim=-1)                          # per‑sample

        return loss                              # reduction later

    # -----------------------------------------------------------------
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch = logits.size(0)

        # Cosine term -------------------------------------------------
        logits_norm = F.normalize(logits, p=2, dim=-1)                   # unit vectors
        target_vec  = F.one_hot(target, num_classes=logits.size(-1)).float()
        cosine_labels = torch.ones(batch, device=logits.device)          # +1 for each pair
        cosine_loss = F.cosine_embedding_loss(
            logits_norm, target_vec, cosine_labels, reduction=self.reduction)

        # Bi‑tempered focal CE ---------------------------------------
        ce = self._bt_ce(logits, target)                                 # no reduction yet

        # Focal weighting – for BTCE we approximate pt as p_{t2}(y)
        with torch.no_grad():
            pt = tempered_softmax(logits, t=self.t2, dim=-1).gather(1, target.unsqueeze(1)).squeeze()

        focal = self.alpha * (1.0 - pt).pow(self.gamma) * ce

        # Reduction ---------------------------------------------------
        if self.reduction == "mean":
            focal = focal.mean()
        elif self.reduction == "sum":
            focal = focal.sum()
        # 'none' → leave per‑sample

        return cosine_loss + self.xent * focal