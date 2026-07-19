"""
Loss functions cho YOLO detection — cho phép swap giữa BCE (mặc định của
Ultralytics) và Focal loss cho phần classification.

Match convention `losses.py` của nhánh Strategy2_TinyImageNet (nơi expose
LOSS_FUNCTION = 'cross_entropy' | 'poly_focal').
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class FocalBCE(nn.Module):
    """
    Focal BCE loss giữ shape output (bs, num_anchors, nc) — drop-in replacement
    cho `nn.BCEWithLogitsLoss(reduction="none")` mà `v8DetectionLoss.__call__`
    sử dụng (line: `self.bce(pred_scores, target_scores).sum() / target_scores_sum`).

    Công thức = BCE với reduction='none' × (1 - p_t)^γ × alpha_factor —
    giống Ultralytics `FocalLoss` NHƯNG không có `.mean(1).sum()` cuối
    (`FocalLoss` gốc trả scalar → không thể thay `self.bce` được).

    Args:
        gamma: focusing parameter — tăng γ → dồn học vào hard examples.
        alpha: balancing parameter — 0 tắt (dùng α cân giữa fg/bg như paper).
    """

    def __init__(self, gamma=1.5, alpha=0.25):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, pred, label):
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        pred_prob = pred.sigmoid()
        p_t = label * pred_prob + (1.0 - label) * (1.0 - pred_prob)
        loss = loss * (1.0 - p_t).pow(self.gamma)
        if self.alpha > 0:
            alpha_factor = label * self.alpha + (1.0 - label) * (1.0 - self.alpha)
            loss = loss * alpha_factor
        return loss  # (bs, num_anchors, nc) — cùng shape với BCE(reduction='none')


def install_cls_loss(yolo_model):
    """
    Override init_criterion() của DetectionModel để dùng loss theo
    Config.LOSS_FUNCTION. Gọi sau `model = YOLO(...)` và TRƯỚC `model.train()`.

    - "bce" : không đụng gì (mặc định của Ultralytics).
    - "focal": thay `criterion.bce` bằng `FocalBCE(FOCAL_GAMMA, FOCAL_ALPHA)`.

    Cơ chế: `BaseModel.loss(batch, preds)` gọi `self.init_criterion()` lazily
    khi criterion chưa tồn tại → override method này sẽ áp dụng cho mọi lượt
    train và val, mà không đụng vào class Ultralytics (safe cho upgrade).
    """
    choice = str(Config.LOSS_FUNCTION).lower()
    if choice == "bce":
        return
    if choice not in ("focal",):
        raise ValueError(
            f"Config.LOSS_FUNCTION={Config.LOSS_FUNCTION!r} không hỗ trợ. "
            "Chọn: 'bce' (Ultralytics default) hoặc 'focal'."
        )

    original_init_criterion = yolo_model.model.init_criterion

    def patched_init_criterion():
        criterion = original_init_criterion()  # v8DetectionLoss(self)
        criterion.bce = FocalBCE(Config.FOCAL_GAMMA, Config.FOCAL_ALPHA)
        return criterion

    yolo_model.model.init_criterion = patched_init_criterion
    print(
        f"  [Loss] Cls loss: FocalBCE (gamma={Config.FOCAL_GAMMA}, "
        f"alpha={Config.FOCAL_ALPHA}) thay cho BCE mặc định"
    )
