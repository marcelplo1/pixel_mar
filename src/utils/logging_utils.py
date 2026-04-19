import time
from collections import defaultdict

import torch
import torch.nn as nn


def _count_params(module):
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def _fmt_params(trainable, total):
    if trainable == total:
        return f"{trainable/1e6:.2f}M"
    return f"{trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total"


def _print_children(module, indent=4):
    """Print parameter counts for each direct child of a module."""
    for name, child in module.named_children():
        t, a = _count_params(child)
        if a == 0:
            continue
        label = f"{name} ({len(child)}x)" if hasattr(child, '__len__') else name
        frozen = f" ({(a-t)/1e6:.2f}M frozen)" if a > t else ""
        print(f"{' '*indent}{label:<36} {_fmt_params(t, a)}{frozen}")


def log_model_parameters(ar_model, denoiser, device, args):
    """Log parameter counts and GFLOPs for all model components."""

    print("\n" + "=" * 60)
    print(" MODEL PARAMETER & COMPUTE BREAKDOWN")
    print("=" * 60)

    # --- AR Model ---
    ar_model_t, ar_model_a = _count_params(ar_model)
    print(f"\n  AR Model: {_fmt_params(ar_model_t, ar_model_a)}")
    _print_children(ar_model)

    # --- Denoiser ---
    den_t, den_a = _count_params(denoiser)
    print(f"\n  Denoiser: {_fmt_params(den_t, den_a)}")
    _print_children(denoiser.denoising_net)

    # --- Total ---
    total = ar_model_t + den_t
    print(f"\n  {'─' * 58}")
    print(f"  TOTAL: {total/1e6:.2f}M trainable")

    # --- Training GFLOPs per image (forward + backward) ---
    try:
        from torch.utils.flop_counter import FlopCounterMode

        N = (args.img_size // args.patch_size) ** 2
        dnet = denoiser.denoising_net
        diff_mul = denoiser.diffusion_batch_mul

        was_training = (ar_model.training, denoiser.training)
        ar_model.train()
        denoiser.train()

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            dummy_orders = torch.randperm(N, device=device).unsqueeze(0)
            dummy_labels = torch.zeros(1, dtype=torch.long, device=device)

            latent_dim = getattr(ar_model, 'latent_dim', None)
            input_dim = latent_dim if latent_dim is not None else args.channels * args.patch_size ** 2
            dummy_ar_model_input = torch.randn(1, N, input_dim, device=device)

            with FlopCounterMode(display=False) as counter:
                out = ar_model(dummy_ar_model_input, dummy_orders, dummy_labels, num_visible=N)
                out_tensor = out[0] if isinstance(out, (tuple, list)) else out
                out_tensor.sum().backward()
            ar_model_gflops = counter.get_total_flops() / 1e9

            # Denoiser processes N * diffusion_batch_mul patches per image during training
            num_patches = N * diff_mul
            D_denoiser = dnet.embedding_dim
            dummy_xt = torch.randn(num_patches, D_denoiser, device=device, requires_grad=True)
            dummy_z = torch.randn(num_patches, dnet.z_proj.in_features, device=device, requires_grad=True)
            dummy_t = torch.rand(num_patches, device=device)

            with FlopCounterMode(display=False) as counter:
                out = dnet(dummy_xt, dummy_z, dummy_t)
                out.sum().backward()
            den_gflops = counter.get_total_flops() / 1e9

        # Clear accumulated dummy grads so they don't contaminate real training
        for p in ar_model.parameters():
            p.grad = None
        for p in denoiser.parameters():
            p.grad = None

        ar_model.train(was_training[0])
        denoiser.train(was_training[1])

        total_gflops = ar_model_gflops + den_gflops
        print(f"\n  Training GFLOPs per image, forward+backward ({N} patches, diffusion_batch_mul={diff_mul}):")
        print(f"AR Model: {ar_model_gflops:>8.2f} GFLOPs")
        print(f"Denoiser: {den_gflops:>8.2f} GFLOPs  ({N}x{diff_mul} = {num_patches} patches)")
        print(f"Total:    {total_gflops:>8.2f} GFLOPs/image")
        print(f"Per step: {total_gflops * args.batch_size:>8.2f} GFLOPs  (batch_size={args.batch_size})")
    except Exception as e:
        print(f"\n  (GFLOPs computation skipped: {e})")

    print("=" * 60 + "\n")


class StepTimer:
    """Tracks per-step wall-clock time for each training stage. Prints and returns wandb-ready stats every `log_interval` steps.
    """

    STAGES = ['data_prep', 'ar_model_total', 'denoiser', 'backward', 'optimizer', 'ema_update']

    def __init__(self, log_interval=50):
        self.timing = defaultdict(float)
        self.steps = 0
        self.log_interval = log_interval
        self._marks = {}

    def mark(self, name):
        """Record a named time point (calls cuda.synchronize)."""
        torch.cuda.synchronize()
        self._marks[name] = time.time()

    def end_step(self):
        """Accumulate timing for this step from marks."""
        m = self._marks
        self.timing['data_prep'] += m['ar_model_start'] - m['step_start']
        self.timing['ar_model_total'] += m['ar_model_end'] - m['ar_model_start']
        self.timing['denoiser'] += m['den_end'] - m['ar_model_end']
        self.timing['backward'] += m['bwd_end'] - m['den_end']
        self.timing['optimizer'] += m['opt_end'] - m['bwd_end']
        self.timing['ema_update'] += m['ema_end'] - m['opt_end']
        self.timing['_total'] += m['ema_end'] - m['step_start']
        self.steps += 1
        self._marks = {}

    def should_log(self):
        return self.steps > 0 and self.steps % self.log_interval == 0

    def print_and_get_stats(self):
        """Print timing table and return dict of stats for wandb."""
        total = self.timing['_total']
        print(f"\n  Timing (avg over {self.steps} steps):")
        for k in self.STAGES:
            v = self.timing[k]
            avg_ms = v / self.steps * 1000
            pct = v / total * 100 if total > 0 else 0
            print(f"    {k:<20} {avg_ms:>8.1f} ms  ({pct:>5.1f}%)")
        print(f"    {'total':<20} {total / self.steps * 1000:>8.1f} ms/step")

        stats = {f"timing/{k}_ms": self.timing[k] / self.steps * 1000 for k in self.STAGES}
        stats["timing/step_total_ms"] = total / self.steps * 1000
        return stats