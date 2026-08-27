"""
EMA và SWA baseline cho Strategy 2 — Image Classification.

Cả hai baseline đều dựng trên ``torch.optim.swa_utils`` của PyTorch — không tự
viết lại phép average, để reviewer đối chiếu thẳng với implementation chính thức:

  - EMA  — Exponential Moving Average of weights (Morales-Brotons et al., TMLR
           2024). ``AveragedModel(multi_avg_fn=get_ema_multi_avg_fn(decay),
           use_buffers=True)``:

               w_ema ← decay · w_ema + (1 − decay) · w_t      (mỗi optimizer step)

           KHÔNG có warmup/ramp cho decay — dùng đúng công thức lũy thừa của thư
           viện. ``use_buffers=True`` ⇒ BN running stats cũng được EMA.

  - SWA  — Stochastic Weight Averaging (Izmailov et al., UAI 2018, Alg. 1).
           ``AveragedModel(multi_avg_fn=get_swa_multi_avg_fn(), use_buffers=False)``:

               w_swa ← (w_swa · n + w_t) / (n + 1)            (mỗi epoch, c = 1)

           ``use_buffers=False`` ⇒ PyTorch average đúng learnable parameters còn
           buffer (BN running stats) được ĐỒNG BỘ từ model nguồn ở mỗi lần update
           — khớp Algorithm 1, trong đó BN stats phải được tính lại bằng một lượt
           forward trên train sau khi average (``evaluate.update_bn``).

Nguyên tắc khách quan
---------------------
1. ``top-k`` / ``last-n`` / ``ema`` chỉ QUAN SÁT weights, KHÔNG đổi LR schedule ⇒
   chung MỘT trajectory ⇒ so sánh PAIRED theo seed.
2. ``swa`` ở mode ``truncate`` giữ LR HẰNG SỐ ở 25% cuối ⇒ ĐỔI trajectory ⇒ phải
   là run RIÊNG. ``main.run_method_legs_if_needed()`` tự tách giúp.
3. Snapshot báo cáo của EMA/SWA là trạng thái ở EPOCH CUỐI — đúng cách dùng chuẩn
   của cả hai paper, không cần nhìn validation nên không có nguy cơ chọn lệch.
4. Sau train, MỌI model đã average (Top-K, Last-N, EMA, SWA) đều đi qua ĐÚNG cùng
   một BN recalibration ``evaluate.update_bn`` rồi mới eval trên test.
"""
import math
import time
from copy import deepcopy

from config import Config


def unwrap_model(model):
    """Bóc ``torch.compile`` / DataParallel để lấy module gốc."""
    seen = set()
    while True:
        if id(model) in seen:
            return model
        seen.add(id(model))
        inner = getattr(model, "_orig_mod", None)
        if inner is None:
            inner = getattr(model, "module", None)
        # ``AveragedModel`` cũng có ``.module`` nên phải chặn để không bóc nhầm.
        if inner is None or not hasattr(inner, "state_dict"):
            return model
        model = inner


def ema_decay_for_window(window_epochs, steps_per_epoch):
    """
    decay sao cho cửa sổ trung bình hiệu dụng ≈ ``window_epochs`` epoch.

    Cửa sổ của EMA là ``1/(1−decay)`` optimizer step, nên::

        decay = 1 − 1 / (window_epochs × steps_per_epoch)

    Tự suy theo số step THẬT của từng dataset là cách duy nhất để ba bộ dữ liệu
    có cửa sổ so sánh được với nhau: CIFAR-100 (351 step/epoch), Tiny ImageNet
    (87 step/epoch) và AgriKD lệch nhau hàng lần, nên một hằng số decay dùng
    chung sẽ cho ba cửa sổ hoàn toàn khác nhau.
    """
    total = max(1.0, float(window_epochs) * float(steps_per_epoch))
    return 1.0 - 1.0 / total


def ema_window_epochs(decay, steps_per_epoch):
    """Cửa sổ trung bình hiệu dụng (≈ 1/(1−decay) step) quy ra epoch."""
    if steps_per_epoch <= 0 or decay >= 1.0:
        return float("inf")
    return (1.0 / (1.0 - float(decay))) / float(steps_per_epoch)


def ema_init_residue(decay, total_updates):
    """
    Tỉ lệ weights KHỞI TẠO còn sót lại trong EMA cuối = ``decay^N`` (không warmup).

    Đây là con số quyết định decay có dùng được không: residue lớn nghĩa là EMA
    vẫn đang kéo theo model pretrained chứ không phải trung bình của trajectory.

    Với decay suy từ cửa sổ W epoch và tổng E epoch thì residue rút gọn thành
    ``e^(−E/W)`` — KHÔNG phụ thuộc dataset. E=60, W=10 ⇒ 0.25%; E=50, W=10 ⇒
    0.67%. Cả hai đều dưới ngưỡng 1% nên grid mặc định an toàn ở mọi bộ dữ liệu.
    """
    if total_updates <= 0:
        return 1.0
    return float(decay) ** int(total_updates)


class _Shadow:
    """Một bản weight-average chạy song song với training (EMA hoặc SWA)."""

    def __init__(self, key, label, kind, avg_model, params, start_epoch=None):
        self.key = key
        self.label = label            # tên hiện trong bảng kết quả / Excel
        self.kind = kind              # "ema" | "swa"
        self.avg = avg_model          # torch.optim.swa_utils.AveragedModel
        self.params = dict(params)
        self.start_epoch = start_epoch   # 1-based, chỉ dùng cho SWA
        self.epochs_seen = []

    @property
    def updates(self):
        return int(self.avg.n_averaged)

    @property
    def ready(self):
        return self.updates > 0

    def update(self, source):
        self.avg.update_parameters(source)


class ShadowAveragingManager:
    """
    Nuôi các shadow EMA/SWA song song với vòng train thuần PyTorch.

    Cách dùng trong ``train_model``::

        mgr = ShadowAveragingManager(model, num_epochs, steps_per_epoch, base_lr, eta_min)
        for epoch in range(1, NUM_EPOCHS + 1):
            lr = mgr.swa_lr_override(epoch)          # constant LR của pha SWA
            if lr is not None:
                for g in optimizer.param_groups:
                    g["lr"] = lr
            train_one_epoch(..., shadow_manager=mgr) # gọi mgr.on_optimizer_step()
            mgr.on_epoch_end(epoch)
            ...
        shadows = mgr.finalize()

    Mọi phép average đều do ``torch.optim.swa_utils`` thực hiện.
    """

    def __init__(self, model, num_epochs, steps_per_epoch, base_lr, eta_min):
        self.enabled = bool(Config.method_enabled("ema") or Config.method_enabled("swa"))
        self.shadows = []
        self.num_epochs = int(num_epochs)
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.total_updates = self.steps_per_epoch * self.num_epochs
        self.base_lr = float(base_lr)
        self.eta_min = float(eta_min)
        self._overhead = 0.0          # giây tiêu tốn riêng cho việc average
        self._steps = 0
        self._swa_start_epoch = None  # 1-based
        self._swa_lr = None

        if not self.enabled:
            return

        source = unwrap_model(model)

        if Config.method_enabled("ema"):
            for decay in self._resolve_decays():
                label = f"EMA (decay {decay:g})"
                self.shadows.append(_Shadow(
                    key=f"ema_{decay:g}".replace(".", "_"),
                    label=label,
                    kind="ema",
                    avg_model=self._build_ema(source, decay),
                    params={
                        "decay": decay,
                        "decay_source": ("auto: cửa sổ "
                                         f"{Config.EMA_WINDOW_EPOCHS}ep × {self.steps_per_epoch} step"
                                         if Config.EMA_DECAYS is None else "đặt tay"),
                        "update_period": 1,
                        "warmup": "none (công thức lũy thừa thuần của PyTorch)",
                        "window_epochs": round(ema_window_epochs(decay, self.steps_per_epoch), 2),
                        "init_residue": round(ema_init_residue(decay, self.total_updates), 6),
                        "impl": "torch.optim.swa_utils.AveragedModel + get_ema_multi_avg_fn",
                    },
                ))

        if Config.method_enabled("swa"):
            mode = Config.swa_lr_mode()
            frac = float(Config.SWA_LR_START_FRAC)
            # 1-based: floor(frac × E) + 1 ⇒ đúng (1−frac) phần epoch cuối.
            start = min(int(math.floor(frac * self.num_epochs)) + 1, self.num_epochs)
            self._swa_start_epoch = start
            if mode != "inherit":
                self._swa_lr = Config.resolved_swa_lr()
            label = (f"SWA (start {frac:.0%}, const LR {self._swa_lr:g})"
                     if self._swa_lr is not None else f"SWA (start {frac:.0%} budget)")
            self.shadows.append(_Shadow(
                key=f"swa_{frac:g}".replace(".", "_"),
                label=label,
                kind="swa",
                avg_model=self._build_swa(source),
                params={
                    "start_frac": frac,
                    "start_epoch": start,
                    "period_epochs": 1,
                    "planned_epochs": self.num_epochs,
                    "lr_schedule": mode,
                    "swa_lr": self._swa_lr,
                    "impl": "torch.optim.swa_utils.AveragedModel + get_swa_multi_avg_fn",
                },
                start_epoch=start,
            ))

        if not self.shadows:
            self.enabled = False
            return

        self._log_setup()

    # ------------------------------------------------------------ construction

    def _resolve_decays(self):
        """decay đặt tay trong Config, hoặc tự suy từ cửa sổ mục tiêu."""
        if Config.EMA_DECAYS:
            return [float(d) for d in Config.EMA_DECAYS]
        return [ema_decay_for_window(Config.EMA_WINDOW_EPOCHS, self.steps_per_epoch)]

    def _build_ema(self, source, decay):
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

        return self._freeze(AveragedModel(
            source, multi_avg_fn=get_ema_multi_avg_fn(float(decay)), use_buffers=True
        ))

    def _build_swa(self, source):
        from torch.optim.swa_utils import AveragedModel, get_swa_multi_avg_fn

        return self._freeze(AveragedModel(
            source, multi_avg_fn=get_swa_multi_avg_fn(), use_buffers=False
        ))

    @staticmethod
    def _freeze(avg):
        avg.module.eval()
        for parameter in avg.module.parameters():
            parameter.grad = None
            parameter.requires_grad_(False)
        return avg

    def _log_setup(self):
        print(f"\n  Weight-averaging baselines (torch.optim.swa_utils) — "
              f"{self.steps_per_epoch} step/epoch × {self.num_epochs} epoch "
              f"= {self.total_updates} optimizer step:")
        for shadow in self.shadows:
            if shadow.kind == "ema":
                residue = shadow.params["init_residue"]
                flag = "  ⚠ QUÁ CAO" if residue > 0.01 else ""
                print(f"    • {shadow.label}: cửa sổ ≈ {shadow.params['window_epochs']:g} epoch"
                      f" | init residue = decay^{self.total_updates} = {residue:.4f}{flag}"
                      f" | {shadow.params['decay_source']}")
            else:
                lr_note = (f"constant LR {shadow.params['swa_lr']:g}"
                           if shadow.params["swa_lr"] is not None
                           else "LR theo scheduler chung (inherit)")
                print(f"    • {shadow.label}: uniform average mỗi epoch, từ epoch "
                      f"{shadow.start_epoch}/{self.num_epochs} ({lr_note})")
        if any(s.kind == "ema" and s.params["init_residue"] > 0.01 for s in self.shadows):
            safe = 0.01 ** (1.0 / max(self.total_updates, 1))
            print(f"    ⚠ init residue > 1%: không có warmup nên EMA vẫn kéo theo weights "
                  f"khởi tạo. Hạ decay xuống ≤ {safe:.6f} cho run {self.total_updates} step này.")

    # ----------------------------------------------------------------- runtime

    def swa_lr_override(self, epoch):
        """
        LR hằng số cần ép cho ``epoch`` (1-based), hoặc None nếu dùng scheduler.

        Đây là phần thuật toán SWA gốc đòi hỏi (Izmailov et al. §3.1: iterate phải
        "explore" vùng phẳng nên LR không được anneal về 0). Ép TRƯỚC khi train
        epoch đó nên LR thực dùng đúng bằng ``Config.resolved_swa_lr()``, bất kể
        scheduler đã step tới đâu.
        """
        if not self.enabled or self._swa_lr is None or self._swa_start_epoch is None:
            return None
        return self._swa_lr if epoch >= self._swa_start_epoch else None

    def on_optimizer_step(self, model):
        """EMA update — gọi ngay sau MỖI ``optimizer.step()``."""
        if not self.enabled:
            return
        self._steps += 1
        ema_shadows = [s for s in self.shadows if s.kind == "ema"]
        if not ema_shadows:
            return
        started = time.time()
        source = unwrap_model(model)
        for shadow in ema_shadows:
            shadow.update(source)
        self._overhead += time.time() - started

    def on_epoch_end(self, epoch, model):
        """SWA snapshot — gọi sau khi train xong epoch (1-based)."""
        if not self.enabled:
            return
        started = time.time()
        source = None
        for shadow in self.shadows:
            if shadow.kind != "swa" or epoch < shadow.start_epoch:
                continue
            if source is None:
                source = unwrap_model(model)
            shadow.update(source)
            shadow.epochs_seen.append(epoch)
        self._overhead += time.time() - started

    # ---------------------------------------------------------------- finalize

    def finalize(self):
        """
        Chốt snapshot: trả list dict mô tả từng shadow, state_dict đã đưa về CPU.

        Returns:
            list[dict] với các khoá: label, kind, state_dict, params, updates,
            epochs_seen, overhead_seconds.
        """
        if not self.enabled or not self.shadows:
            return []

        started = time.time()
        out = []
        for shadow in self.shadows:
            if not shadow.ready:
                print(f"  ⚠ {shadow.label}: chưa gom được snapshot nào "
                      f"(train dừng sớm?) — bỏ qua")
                continue
            state = {
                key: value.detach().to("cpu", copy=True)
                for key, value in shadow.avg.module.state_dict().items()
            }
            out.append({
                "label": shadow.label,
                "kind": shadow.kind,
                "state_dict": state,
                "params": dict(shadow.params),
                "updates": shadow.updates,
                "epochs_seen": list(shadow.epochs_seen) or None,
            })
            print(f"  ✓ {shadow.label}: {shadow.updates} update"
                  + (f", epoch {shadow.epochs_seen[0]}–{shadow.epochs_seen[-1]}"
                     if shadow.epochs_seen else ""))
        self._overhead += time.time() - started

        # Shadow không còn cần thiết — giải phóng VRAM trước khi eval.
        share = self._overhead / max(len(out), 1)
        for entry in out:
            entry["overhead_seconds"] = share
        for shadow in self.shadows:
            shadow.avg = None
        self.shadows = []
        return out

    @property
    def overhead_seconds(self):
        """Tổng giây tiêu tốn riêng cho việc nuôi shadow (ngoài chi phí train)."""
        return self._overhead
