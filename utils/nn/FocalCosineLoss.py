import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalCosineLoss(nn.Module):
    """
    Implementation Focal cosine loss from the "Data-Efficient Deep Learning Method for Image Classification.

    Using Data Augmentation, Focal Cosine Loss, and Ensemble" (https://arxiv.org/abs/2007.07805).
    Credit: https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271
    """
    def __init__(self, alpha=1, gamma=2, xent=.1):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.xent = xent

        self.y = torch.Tensor([1]).cuda()

    def forward(self, input, target, reduction="mean"):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y, reduction=reduction)

        cent_loss = F.cross_entropy(F.normalize(input), target, reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * cent_loss

        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss

### Another approach
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalCosineLoss(nn.Module):
    """
    Implementation Focal cosine loss from the "Data-Efficient Deep Learning Method for Image Classification.
    Inspired by:
    Using Data Augmentation, Focal Cosine Loss, and Ensemble" (https://arxiv.org/abs/2007.07805).
    Credit: https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271

    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0,
                 xent: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.alpha, self.gamma, self.xent, self.reduction = alpha, gamma, xent, reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # --- Cosine part -----------------------------------------------------
        target_onehot = F.one_hot(target, num_classes=logits.size(-1)).float()
        logits_norm   = F.normalize(logits, p=2, dim=-1)          # unit‑norm
        cosine_loss   = F.cosine_embedding_loss(
            logits_norm, target_onehot,
            torch.ones(logits.size(0), device=logits.device),     # shape = (batch,)
            reduction=self.reduction)

        # --- Focal cross‑entropy part ---------------------------------------
        ce_loss = F.cross_entropy(logits, target, reduction="none")   # keep per‑sample
        pt      = torch.exp(-ce_loss)                                 # pt = softmax prob of truth
        focal   = self.alpha * (1.0 - pt).pow(self.gamma) * ce_loss   # elementwise

        if self.reduction == "mean":
            focal = focal.mean()
        elif self.reduction == "sum":
            focal = focal.sum()
        # else ('none') keep as‑is

        return cosine_loss + self.xent * focal
