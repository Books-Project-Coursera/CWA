"""
EMA và SWA baseline cho Strategy 2 — Object Detection (Ultralytics YOLO).

Cả hai baseline đều dựng trên ``torch.optim.swa_utils`` của PyTorch — không tự
viết lại phép average, để reviewer đối chiếu thẳng với implementation chính
thức:

  - EMA  — Exponential Moving Average of weights (Morales-Brotons et al., TMLR
           2024). ``AveragedModel(multi_avg_fn=get_ema_multi_avg_fn(decay),
           use_buffers=True)``:

               w_ema ← decay · w_ema + (1 − decay) · w_t      (mỗi optimizer step)

           KHÔNG có warmup/ramp cho decay — dùng đúng công thức lũy thừa của
           thư viện. ``use_buffers=True`` ⇒ BN running stats cũng được EMA,
           giống ``ultralytics.utils.torch_utils.ModelEMA``.

           ⚠ Hệ quả của việc không warmup: phần khởi tạo còn sót lại trong EMA
           cuối là ``decay^N`` với N = tổng số optimizer step. Với N ≈ 11.7k
           (VOC) thì decay = 0.9999 để lại 31 % là weights ban đầu ⇒ EMA hỏng.
           ``on_train_start`` in ra residue của TỪNG decay để bạn kiểm tra ngay
           từ log; xem ``Config.EMA_DECAYS`` để biết grid an toàn.

  - SWA  — Stochastic Weight Averaging (Izmailov et al., UAI 2018, Alg. 1).
           ``AveragedModel(multi_avg_fn=get_swa_multi_avg_fn(), use_buffers=False)``:

               w_swa ← (w_swa · n + w_t) / (n + 1)             (mỗi epoch, c = 1)

           ``use_buffers=False`` ⇒ PyTorch average đúng learnable parameters và
           ĐỒNG BỘ buffer (BN running stats) từ model nguồn ở mỗi lần update —
           khớp Algorithm 1, trong đó BN stats phải được tính lại bằng một lượt
           forward trên train sau khi average (``evaluate.update_bn_stats``).

           SWA gốc còn đòi constant/cyclic LR ở pha cuối. Bật bằng
           ``Config.SWA_LR_SCHEDULE = "constant"`` — xem ``_install_swa_lr_schedule``.

Nguyên tắc khách quan
---------------------
1. Trong MỘT run, cả ba method (Top-K, EMA, SWA) đọc weights từ CÙNG một
   trajectory. EMA/SWA chỉ QUAN SÁT weights, không can thiệp optimization
   ("SWA does not affect optimization since the averaged model is not used in
   the training loop") ⇒ so sánh PAIRED theo seed.
   NGOẠI LỆ: khi ``SWA_LR_SCHEDULE = "constant"`` thì LR schedule đổi ⇒ đó là
   một run KHÁC; run đó vẫn so paired được nội bộ giữa ba method.
2. Validation trong lúc train luôn dùng RAW weights (TopKCheckpointManager tắt
   ``trainer.ema``) nên fitness / best.pt / early stopping giống hệt nhau ở mọi
   tổ hợp ``--method``.
3. Snapshot báo cáo của EMA/SWA mặc định là trạng thái ở EPOCH CUỐI
   (``SHADOW_SELECT = "final"``) — đúng cách dùng chuẩn của cả hai paper, và
   không cần nhìn validation. Bật ``SHADOW_VAL_ENABLED`` nếu muốn thêm đường
   cong val của shadow (chỉ để phụ lục) hoặc chọn snapshot theo val.
4. Nếu bật val shadow, lượt val đó KHÔNG được làm lệch trajectory: RNG state
   (python/numpy/torch/cuda) được lưu–khôi phục, và shadow được val qua một bản
   sao nên ``model.half()`` của validator không đụng weights FP32.
5. Sau train, mọi model đã average đi qua ĐÚNG cùng một BN recalibration, và
   baseline có dòng control "Strategy 1 + BN recal".
"""
import json
import math
import random
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from config import Config

# Manifest mô tả các shadow đã lưu; đặt ở GỐC run dir (không phải weights/) để
# không bị delete_checkpoint_artifacts() xóa cùng checkpoint tạm.
SHADOW_MANIFEST = "averaging_shadows.json"


def _slug(text):
    """'EMA (decay 0.999)' → 'ema_decay_0_999' — dùng đặt tên file .pt."""
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()
    return slug or "shadow"


def _unwrap(model):
    """Bóc DP/DDP/torch.compile — dùng lại helper đã có trong train.py."""
    from train import unwrap_ultralytics_model

    return unwrap_ultralytics_model(model)


def ema_window_epochs(decay, steps_per_epoch):
    """Cửa sổ trung bình hiệu dụng của EMA (≈ 1/(1−decay) step) quy ra epoch."""
    if steps_per_epoch <= 0 or decay >= 1.0:
        return float("inf")
    return (1.0 / (1.0 - float(decay))) / float(steps_per_epoch)


def ema_init_residue(decay, total_updates):
    """
    Tỉ lệ weights KHỞI TẠO còn sót lại trong EMA cuối = decay^N (không warmup).

    Đây là con số quyết định decay nào dùng được: residue lớn nghĩa là EMA vẫn
    đang kéo theo model pretrained chứ không phải trung bình của trajectory.
    Giữ dưới ~1 % thì decay mới có ý nghĩa.
    """
    if total_updates <= 0:
        return 1.0
    return float(decay) ** int(total_updates)


class _Shadow:
    """Một bản weight-average chạy song song với training (EMA hoặc SWA)."""

    def __init__(self, key, label, short, kind, avg_model, params):
        self.key = key            # slug → tên file .pt
        self.label = label        # tên đầy đủ hiện trong bảng kết quả
        self.short = short        # nhãn ngắn cho chart
        self.kind = kind          # "ema" | "swa"
        self.avg = avg_model      # torch.optim.swa_utils.AveragedModel
        self.params = dict(params)
        self.epochs_seen = []     # epoch nào đã được đưa vào average (SWA)
        self.history = []         # [(epoch, val fitness)] — chỉ khi bật val shadow
        self.best = None          # {"epoch", "fitness", "state"}
        self.final = None         # {"epoch", "fitness", "state"}

    @property
    def module(self):
        """Model mang weight đã average (nn.Module thuần, dùng được với YOLO)."""
        return self.avg.module

    @property
    def updates(self):
        return int(self.avg.n_averaged)

    @property
    def ready(self):
        return self.updates > 0

    def update(self, raw):
        """Một bước average — toàn bộ phép toán do PyTorch thực hiện."""
        self.avg.update_parameters(raw)


class ShadowAveragingManager:
    """
    Callback bundle: nuôi các shadow EMA/SWA bằng ``torch.optim.swa_utils``,
    (tuỳ chọn) chấm điểm trên val', rồi ghi snapshot ra <run_dir>/weights/ kèm
    manifest ở gốc run dir.

    Đăng ký:
        manager = ShadowAveragingManager()
        model.add_callback("on_train_start",      manager.on_train_start)
        model.add_callback("on_train_epoch_end",  manager.on_train_epoch_end)
        model.add_callback("on_fit_epoch_end",    manager.on_fit_epoch_end)
        model.add_callback("on_train_end",        manager.on_train_end)
    """

    def __init__(self):
        self.shadows = []
        self.enabled = bool(Config.method_enabled("ema") or Config.method_enabled("swa"))
        self.select = str(Config.SHADOW_SELECT).lower()
        self.val_period = max(1, int(Config.SHADOW_VAL_PERIOD))
        self.val_enabled = bool(Config.SHADOW_VAL_ENABLED)
        if not self.val_enabled:
            # Không chấm điểm trên val' thì chỉ còn snapshot cuối là hợp lệ.
            self.select = "final"
        self._swa_start_epoch = {}
        self._steps_per_epoch = 0
        self._total_updates = 0
        self._last_val_epoch = -1

    # ---------------------------------------------------------------- setup

    def on_train_start(self, trainer):
        """Dựng shadow, gắn EMA vào optimizer step, và (tuỳ chọn) đổi LR schedule."""
        if not self.enabled:
            return

        # DDP: chỉ rank chính có validator + chạy on_train_end, nên chỉ rank đó
        # nuôi shadow (weights đã được đồng bộ giữa các rank sau mỗi step).
        from ultralytics.utils import RANK

        if RANK not in (-1, 0):
            self.enabled = False
            return

        if int(getattr(trainer, "start_epoch", 0) or 0) > 0:
            print(
                "  ⚠ Resume: EMA/SWA shadow bắt đầu lại từ epoch "
                f"{trainer.start_epoch} nên KHÔNG phản ánh toàn bộ trajectory. "
                "Chạy run mới nếu cần số liệu EMA/SWA cho báo cáo."
            )

        raw = _unwrap(trainer.model)
        # Cửa sổ EMA và residue phải tính theo số OPTIMIZER STEP, không phải số
        # batch: Ultralytics gradient-accumulate ``nbs / batch`` batch cho mỗi
        # step (batch=128, nbs=64 ⇒ accumulate=1, nhưng batch nhỏ thì khác hẳn).
        try:
            batches_per_epoch = max(1, len(trainer.train_loader))
        except TypeError:
            batches_per_epoch = 1
        accumulate = max(1, int(getattr(trainer, "accumulate", 1) or 1))
        self._steps_per_epoch = max(1, batches_per_epoch // accumulate)
        epochs = int(getattr(trainer, "epochs", Config.EPOCHS) or Config.EPOCHS)
        self._total_updates = self._steps_per_epoch * epochs

        # LR schedule của SWA phải cài TRƯỚC khi in log để con số hiện ra đúng.
        swa_lr_start = None
        swa_mode = Config.swa_lr_mode() if Config.method_enabled("swa") else "inherit"
        if swa_mode != "inherit":
            swa_lr_start = self._install_swa_lr_schedule(trainer, epochs)

        if Config.method_enabled("ema"):
            for decay in Config.EMA_DECAYS:
                decay = float(decay)
                label = f"EMA (decay {decay:g})"
                self.shadows.append(_Shadow(
                    key=_slug(label),
                    label=label,
                    short=f"EMA d={decay:g}",
                    kind="ema",
                    avg_model=self._build_ema(raw, decay),
                    params={
                        "decay": decay,
                        "update_period": int(Config.EMA_UPDATE_PERIOD),
                        "warmup": "none (công thức lũy thừa thuần của PyTorch)",
                        "window_epochs": round(ema_window_epochs(decay, self._steps_per_epoch), 2),
                        "init_residue": round(ema_init_residue(decay, self._total_updates), 6),
                        "impl": "torch.optim.swa_utils.AveragedModel + get_ema_multi_avg_fn",
                    },
                ))

        if Config.method_enabled("swa"):
            # Cửa sổ average PHẢI trùng pha constant LR — average trên đoạn còn đang
            # anneal là sai thuật toán. Nên ở mode 'extend'/'truncate', mốc bắt đầu
            # lấy thẳng từ _install_swa_lr_schedule() chứ không dùng SWA_START_FRACS.
            starts = ([swa_lr_start] if swa_lr_start is not None
                      else [min(max(int(math.floor(float(f) * epochs)), 0), max(epochs - 1, 0))
                            for f in Config.SWA_START_FRACS])
            for start_epoch in starts:
                frac = start_epoch / epochs if epochs else 0.0
                # Ghi luôn đặc điểm trajectory vào TÊN dòng: bảng kết quả của run này
                # rồi sẽ đứng cạnh bảng của run chuẩn trong paper, phải nhìn là biết
                # ngay dòng nào đến từ trajectory nào.
                if swa_mode == "extend":
                    budget = epochs / max(int(Config.EPOCHS), 1)
                    label = (f"SWA ({budget:.2f} budget, const LR "
                             f"{Config.resolved_swa_lr():g})")
                    short = f"SWA {budget:.2f}b constLR"
                elif swa_mode == "truncate":
                    label = f"SWA (start {frac:.0%}, const LR {Config.resolved_swa_lr():g})"
                    short = f"SWA@{frac:.0%} constLR"
                else:
                    label = f"SWA (start {frac:.0%} budget)"
                    short = f"SWA@{frac:.0%}"
                shadow = _Shadow(
                    key=_slug(label),
                    label=label,
                    short=short,
                    kind="swa",
                    avg_model=self._build_swa(raw),
                    params={
                        "start_frac": frac,
                        "start_epoch": start_epoch,
                        "period_epochs": int(Config.SWA_PERIOD),
                        "planned_epochs": epochs,
                        "lr_schedule": swa_mode,
                        "cosine_epochs": int(Config.EPOCHS) if swa_mode == "extend" else None,
                        "total_epochs": epochs,
                        "budget": round(epochs / max(int(Config.EPOCHS), 1), 4),
                        "swa_lr": Config.resolved_swa_lr() if swa_lr_start is not None else None,
                        "swa_lr_source": ("auto (lr0+lr0*lrf)/2" if Config.SWA_LR is None
                                          else "explicit") if swa_lr_start is not None else None,
                        "impl": "torch.optim.swa_utils.AveragedModel + get_swa_multi_avg_fn",
                    },
                )
                self._swa_start_epoch[shadow.key] = start_epoch
                self.shadows.append(shadow)

        if not self.shadows:
            self.enabled = False
            return

        self._install_optimizer_hook(trainer)
        self._log_setup(epochs)

    def _log_setup(self, epochs):
        print("  Weight-averaging baselines (torch.optim.swa_utils):")
        for shadow in self.shadows:
            if shadow.kind == "ema":
                residue = shadow.params["init_residue"]
                flag = "  ⚠ QUÁ CAO" if residue > 0.01 else ""
                print(
                    f"    • {shadow.label}: update mỗi {shadow.params['update_period']} "
                    f"optimizer step, không warmup | cửa sổ ≈ "
                    f"{shadow.params['window_epochs']:g} epoch | "
                    f"init residue = decay^{self._total_updates} = {residue:.4f}{flag}"
                )
            else:
                lr_note = (f", constant LR = {shadow.params['swa_lr']:g}"
                           if shadow.params.get("swa_lr") else ", LR theo schedule chung")
                print(
                    f"    • {shadow.label}: uniform average mỗi "
                    f"{shadow.params['period_epochs']} epoch, từ epoch "
                    f"{shadow.params['start_epoch']} (0-based) / {epochs}{lr_note}"
                )
        print(
            f"    Snapshot báo cáo: {self.select}"
            + (f" (val shadow mỗi {self.val_period} epoch)" if self.val_enabled
               else " (không val shadow — không tốn thêm thời gian train)")
        )
        if any(s.kind == "ema" and s.params["init_residue"] > 0.01 for s in self.shadows):
            print(
                "    ⚠ init residue > 1%: không có warmup nên EMA vẫn đang kéo theo "
                "weights khởi tạo. Hạ decay trong Config.EMA_DECAYS "
                f"(an toàn: decay ≤ {0.01 ** (1.0 / max(self._total_updates, 1)):.5f} "
                f"cho run {self._total_updates} step này)."
            )

    # --------------------------------------------------- torch.optim.swa_utils

    def _clean_copy_source(self, raw):
        """
        Tạm gỡ ``criterion`` khỏi model nguồn trước khi ``AveragedModel`` deepcopy.

        ``BaseModel.loss()`` gắn criterion (giữ tensor GPU + class custom như
        FocalBCE) vào model; deepcopy cả nó làm shadow phình ra vô ích.
        """
        return getattr(raw, "criterion", None)

    def _build_ema(self, raw, decay):
        """
        EMA shadow = AveragedModel với hàm cập nhật EMA chính chủ của PyTorch.

        ``use_buffers=True``: average cả BN running stats — đúng ngữ nghĩa EMA
        thường dùng (và giống ModelEMA của Ultralytics).
        """
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

        criterion = self._clean_copy_source(raw)
        raw.criterion = None
        try:
            avg = AveragedModel(
                raw, multi_avg_fn=get_ema_multi_avg_fn(float(decay)), use_buffers=True
            )
        finally:
            raw.criterion = criterion
        return self._finalize(avg)

    def _build_swa(self, raw):
        """
        SWA shadow = AveragedModel mặc định (trung bình cộng đều).

        ``use_buffers=False``: PyTorch average đúng parameters, còn buffer được
        ĐỒNG BỘ từ model nguồn ở mỗi update — khớp Algorithm 1 của Izmailov, nơi
        BN stats được tính lại sau khi average chứ không phải lấy trung bình.
        """
        from torch.optim.swa_utils import AveragedModel, get_swa_multi_avg_fn

        criterion = self._clean_copy_source(raw)
        raw.criterion = None
        try:
            avg = AveragedModel(raw, multi_avg_fn=get_swa_multi_avg_fn(), use_buffers=False)
        finally:
            raw.criterion = criterion
        return self._finalize(avg)

    @staticmethod
    def _finalize(avg):
        avg.module.criterion = None
        avg.module.eval()
        for parameter in avg.module.parameters():
            parameter.grad = None
            parameter.requires_grad_(False)
        return avg

    # ------------------------------------------------------- SWA LR schedule

    def _install_swa_lr_schedule(self, trainer, epochs):
        """
        Cài pha constant-LR của SWA. Trả về epoch bắt đầu pha đó.

        Hai mode (Izmailov et al. mô tả cả hai):

        - ``extend``  : cosine chạy TRỌN ``Config.EPOCHS`` epoch — đúng bằng run
          chuẩn của Top-K/EMA, LR anneal hết xuống ``lr0·lrf`` — rồi MỚI nối thêm
          ``swa_extra_epochs()`` epoch constant LR. "We use a full budget to get an
          initialization for the procedure, and then train ... for 0.25 and 0.5
          budgets" (§4.1). ``EPOCHS`` epoch đầu trùng khít run chuẩn.

          Lưu ý: ``_setup_scheduler`` của Ultralytics dựng cosine trên TỔNG số
          epoch (đã cộng phần nối thêm), nên phải DỰNG LẠI cosine trên đúng
          ``Config.EPOCHS`` thì 100 epoch đầu mới khớp run chuẩn.

        - ``truncate``: giữ nguyên cosine của Ultralytics nhưng CẮT ở
          ``SWA_LR_START_FRAC × EPOCHS`` rồi chạy nốt bằng constant — tổng budget
          giữ nguyên ("1 budget", §3.2).

        Cách cài — đổi ĐÚNG lambda của scheduler có sẵn, không dựng scheduler mới
        nên không đụng ``initial_lr`` của các param group::

            LR(epoch) = lr0 · lf(epoch),   lf = LambdaLR.lr_lambdas[i]

        ``trainer.lf`` cũng được cập nhật vì warmup của Ultralytics dùng nó làm đích
        nội suy (chỉ ảnh hưởng ``warmup_epochs`` đầu, nằm ngoài pha SWA).
        """
        from ultralytics.utils.torch_utils import one_cycle

        mode = Config.swa_lr_mode()
        lr0 = float(getattr(trainer.args, "lr0", Config.LR0))
        # SWA_LR = None ⇒ tự tính (lr0 + lr0·lrf)/2 — "intermediate value between the
        # largest and the smallest learning rate used in the annealing scheme"
        # (Izmailov et al. §4.3). Xem Config.resolved_swa_lr().
        swa_lr = Config.resolved_swa_lr()
        ratio = swa_lr / lr0

        if mode == "extend":
            cosine_epochs = int(Config.EPOCHS)
            start_epoch = min(cosine_epochs, max(epochs - 1, 0))
            lrf = float(getattr(trainer.args, "lrf", Config.LRF))
            if bool(getattr(trainer.args, "cos_lr", Config.COS_LR)):
                base_lf = one_cycle(1, lrf, cosine_epochs)
            else:
                base_lf = (lambda x: max(1 - x / cosine_epochs, 0) * (1.0 - lrf) + lrf)
            phase = (f"cosine TRỌN {cosine_epochs} epoch (trùng khít run chuẩn) + "
                     f"{epochs - cosine_epochs} epoch constant ⇒ {epochs} epoch "
                     f"({epochs / cosine_epochs:.2f} budget)")
        else:
            start_epoch = min(
                max(int(math.floor(float(Config.SWA_LR_START_FRAC) * epochs)), 0),
                max(epochs - 1, 0),
            )
            base_lf = trainer.lf
            phase = (f"cắt cosine ở epoch {start_epoch}/{epochs} rồi constant "
                     f"⇒ tổng vẫn {epochs} epoch (1.00 budget)")

        def swa_lf(x):
            return base_lf(x) if x < start_epoch else ratio

        trainer.lf = swa_lf
        scheduler = getattr(trainer, "scheduler", None)
        lambdas = getattr(scheduler, "lr_lambdas", None)
        if lambdas is None:
            raise RuntimeError(
                "Không tìm thấy LambdaLR.lr_lambdas — không thể cài constant LR cho SWA. "
                "Kiểm tra lại phiên bản Ultralytics/PyTorch."
            )
        scheduler.lr_lambdas = [swa_lf] * len(lambdas)

        source = "tự tính (lr0 + lr0·lrf)/2" if Config.SWA_LR is None else "đặt tay"
        print(
            f"  SWA LR schedule [{mode}]: {phase}; constant LR = {swa_lr:g} "
            f"({ratio:.3f} × lr0, {source}), bắt đầu từ epoch {start_epoch}"
        )
        return start_epoch

    # ------------------------------------------------------------- updates

    def _install_optimizer_hook(self, trainer):
        """
        Bọc ``trainer.optimizer_step`` để EMA update đúng MỖI optimizer step.

        Dùng optimizer_step chứ không dùng callback ``on_train_batch_end`` vì
        Ultralytics chỉ step khi đủ ``accumulate`` batch — bám vào batch sẽ đếm
        sai số update.
        """
        ema_shadows = [s for s in self.shadows if s.kind == "ema"]
        if not ema_shadows:
            return

        original_step = trainer.optimizer_step
        state = {"steps": 0}

        def optimizer_step():
            original_step()
            state["steps"] += 1
            raw = None
            for shadow in ema_shadows:
                if state["steps"] % int(shadow.params["update_period"]):
                    continue
                if raw is None:
                    raw = _unwrap(trainer.model)
                shadow.update(raw)

        trainer.optimizer_step = optimizer_step

    def on_train_epoch_end(self, trainer):
        """SWA lấy snapshot sau optimizer step CUỐI của epoch, trước validation."""
        if not self.enabled:
            return
        epoch = int(trainer.epoch)
        raw = None
        for shadow in self.shadows:
            if shadow.kind != "swa":
                continue
            start = self._swa_start_epoch[shadow.key]
            if epoch < start:
                continue
            if (epoch - start) % max(1, int(shadow.params["period_epochs"])):
                continue
            if raw is None:
                raw = _unwrap(trainer.model)
            shadow.update(raw)
            shadow.epochs_seen.append(epoch)

    # ---------------------------------------------------- validation on val'

    def on_fit_epoch_end(self, trainer):
        """
        (Tuỳ chọn) chấm điểm shadow trên val' — CÙNG dataloader, CÙNG fitness mà
        Ultralytics vừa dùng cho raw model ở epoch này.

        Mặc định TẮT: snapshot báo cáo là trạng thái ở epoch cuối nên không cần
        validation. Bật khi muốn đường cong val của shadow cho phụ lục, hoặc khi
        ``SHADOW_SELECT = "best_val"``.
        """
        if not self.enabled or not self.val_enabled:
            return

        validator = getattr(trainer, "validator", None)
        if validator is None:
            return
        # ``final_eval()`` gọi LẠI on_fit_epoch_end sau khi validate best.pt ở
        # chế độ standalone (validator.training = False). Chạy shadow ở đó vừa
        # thừa một lượt val, vừa ghi đè ``validator.metrics`` — chính là thứ
        # ``model.train()`` trả về và được in ra dưới nhãn "FINAL VALIDATION on
        # val' (best ckpt)". Bỏ qua đúng lần gọi đó.
        if not getattr(validator, "training", True):
            return

        epoch = int(trainer.epoch)
        if epoch == self._last_val_epoch:
            return  # phòng thêm trường hợp callback được gọi 2 lần trong 1 epoch

        final_epoch = epoch + 1 >= int(getattr(trainer, "epochs", 0) or 0)
        due = (epoch % self.val_period == 0) or final_epoch or bool(getattr(trainer, "stop", False))
        pending = [s for s in self.shadows if s.ready]
        if not pending or not due:
            return

        self._last_val_epoch = epoch
        for shadow in pending:
            fitness = self._validate_shadow(trainer, shadow)
            if fitness is None:
                continue
            shadow.history.append((epoch, fitness))
            if shadow.best is None or fitness > shadow.best["fitness"]:
                shadow.best = {
                    "epoch": epoch,
                    "fitness": fitness,
                    "state": self._capture(shadow),
                }
            print(f"      [{shadow.short}] val' fitness epoch {epoch}: {fitness:.6f}"
                  + ("  ← best" if shadow.best["epoch"] == epoch else ""))

    def _validate_shadow(self, trainer, shadow):
        """
        Val một BẢN SAO của shadow, giữ nguyên mọi state của trainer.

        - Val trên bản sao vì ``BaseValidator.__call__`` gọi ``model.half()``
          khi train có AMP; làm trực tiếp lên shadow sẽ hạ weights đang tích
          lũy xuống FP16.
        - ``trainer.ema.ema`` là model mà validator lấy khi ``training=True``
          nên chỉ cần tráo tạm rồi trả lại.
        - RNG state được lưu/khôi phục ⇒ lượt val này KHÔNG THỂ làm lệch
          trajectory của training.
        """
        import torch

        validator = getattr(trainer, "validator", None)
        if validator is None:
            return None

        rng = self._rng_state()
        saved_model = trainer.ema.ema
        saved_plots = getattr(validator.args, "plots", False)
        probe = None
        try:
            probe = deepcopy(shadow.module).eval()
            trainer.ema.ema = probe
            validator.args.plots = False  # không ghi đè val_batch*.jpg của raw
            stats = validator(trainer)
            fitness = stats.get("fitness")
            return float(fitness) if fitness is not None and math.isfinite(float(fitness)) else None
        except Exception as exc:  # val shadow hỏng không được giết cả run
            print(f"      ⚠ Bỏ qua val shadow {shadow.short}: {type(exc).__name__}: {exc}")
            return None
        finally:
            trainer.ema.ema = saved_model
            validator.args.plots = saved_plots
            del probe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._restore_rng(rng)

    @staticmethod
    def _rng_state():
        import numpy as np
        import torch

        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _restore_rng(state):
        import numpy as np
        import torch

        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

    @staticmethod
    def _capture(shadow):
        """Chụp state_dict của shadow về CPU (giữ trong RAM, chưa ghi disk)."""
        return {
            key: value.detach().to("cpu", copy=True)
            for key, value in shadow.module.state_dict().items()
        }

    # ------------------------------------------------------------ finalize

    def on_train_end(self, trainer):
        """Ghi snapshot đã chọn của từng shadow ra .pt + manifest JSON."""
        if not self.enabled or not self.shadows:
            return
        import torch
        import ultralytics

        run_dir = Path(trainer.save_dir)
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        last_epoch = int(trainer.epoch)

        manifest = {}
        for shadow in self.shadows:
            if not shadow.ready:
                print(f"  ⚠ {shadow.label}: chưa gom được snapshot nào "
                      f"(train dừng ở epoch {last_epoch}) — bỏ qua")
                continue

            shadow.final = {
                "epoch": last_epoch,
                "fitness": shadow.history[-1][1] if shadow.history else None,
                "state": self._capture(shadow),
            }
            chosen = shadow.best if (self.select == "best_val" and shadow.best) else shadow.final
            selection = "best_val" if chosen is shadow.best else "final"
            # SWA: chỉ những epoch ĐÃ nằm trong average tại thời điểm snapshot
            # mới được ghi nhận — nếu chọn best_val ở epoch e thì các epoch sau
            # e chưa hề tham gia.
            averaged_epochs = (
                [e for e in shadow.epochs_seen if e <= int(chosen["epoch"])]
                if shadow.kind == "swa" else None
            )

            module = deepcopy(shadow.module).cpu().float()
            module.load_state_dict(chosen["state"], strict=True)
            module.criterion = None
            module.eval()

            filename = f"avg_{shadow.key}.pt"
            torch.save(
                {
                    "epoch": int(chosen["epoch"]),
                    "best_fitness": chosen["fitness"],
                    "model": module,
                    "ema": None,
                    "updates": int(shadow.updates),
                    "optimizer": None,
                    "train_args": vars(trainer.args),
                    "train_metrics": {"fitness": chosen["fitness"]},
                    "date": datetime.now().isoformat(timespec="seconds"),
                    "version": ultralytics.__version__,
                    "weight_source": f"{shadow.kind}_shadow",
                    "average_type": (
                        "torch.optim.swa_utils EMA" if shadow.kind == "ema"
                        else "torch.optim.swa_utils uniform average"
                    ),
                    "method_params": shadow.params,
                    "bn_recalibrated": False,
                },
                str(weights_dir / filename),
            )

            manifest[shadow.key] = {
                "file": filename,
                "method": shadow.kind,
                "label": shadow.label,
                "short": shadow.short,
                "params": shadow.params,
                "selection": selection,
                "epoch": int(chosen["epoch"]),
                "val_fitness": chosen["fitness"],
                "final_epoch": int(shadow.final["epoch"]),
                "final_val_fitness": shadow.final["fitness"],
                "best_val_epoch": shadow.best["epoch"] if shadow.best else None,
                "best_val_fitness": shadow.best["fitness"] if shadow.best else None,
                "updates": int(shadow.updates),
                "averaged_epochs": averaged_epochs or None,
                "averaged_count": len(averaged_epochs) if averaged_epochs else None,
                "val_history": [[int(e), float(f)] for e, f in shadow.history],
            }
            # State đã ghi ra disk — trả RAM lại ngay, mỗi snapshot là 1 model.
            shadow.best = shadow.final = None
            fitness_txt = (f"{chosen['fitness']:.6f}"
                           if chosen["fitness"] is not None else "n/a (không val shadow)")
            print(
                f"  ✓ {shadow.label} → weights/{filename} "
                f"({shadow.updates} update, chọn theo {selection}, "
                f"epoch {chosen['epoch']}, val fitness {fitness_txt})"
            )

        if manifest:
            (run_dir / SHADOW_MANIFEST).write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        # Shadow không còn cần thiết — giải phóng VRAM trước khi eval.
        for shadow in self.shadows:
            shadow.avg = None
        self.shadows = []


def load_shadow_manifest(run_dir):
    """
    Đọc manifest shadow của một run.

    Returns:
        list[dict] theo thứ tự EMA trước, SWA sau; rỗng nếu run không có shadow.
    """
    path = Path(run_dir) / SHADOW_MANIFEST
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = list(data.values())
    order = {"ema": 0, "swa": 1}
    entries.sort(key=lambda e: (order.get(e.get("method"), 9), str(e.get("label"))))
    return entries
