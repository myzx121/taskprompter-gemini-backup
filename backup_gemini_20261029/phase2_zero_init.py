"""Backup-only Phase 2 invariants for TaskPrompter.

This file does not touch the formal queues.  It validates the strict-zero-init
building blocks before wiring them into the expensive NYUD runner.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn


class ZeroFiLM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return (1.0 + self.gamma.view(*shape)) * x + self.beta.view(*shape)


class ZeroLoRA(nn.Module):
    def __init__(self, channels: int, rank: int = 8):
        super().__init__()
        self.lora_A = nn.Conv2d(channels, rank, 1, bias=False)
        self.lora_B = nn.Conv2d(rank, channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return x + self.lora_B(self.lora_A(x))


class DecoupledCalibration(nn.Module):
    """Task-output calibration; never receives shared features directly."""
    def __init__(self, channels: int, tasks):
        super().__init__()
        self.tasks = tuple(tasks)
        self.log_scale = nn.ParameterDict({t: nn.Parameter(torch.zeros(1)) for t in tasks})
        self.bias = nn.ParameterDict({t: nn.Parameter(torch.zeros(channels)) for t in tasks})

    def forward(self, raw, task):
        scale = self.log_scale[task].exp()
        bias = self.bias[task].view(1, -1, 1, 1)
        return raw * scale + bias


class DualPathGate(nn.Module):
    def __init__(self, channels: int, tasks):
        super().__init__()
        self.tasks = tuple(tasks)
        prior = {"depth": 1.5, "normals": 1.5, "semseg": 0.0, "edge": 0.0}
        self.alpha = nn.ParameterDict({t: nn.Parameter(torch.full((channels,), prior[t])) for t in tasks})
        self.charge_norm = nn.LayerNorm(channels)
        self.event_norm = nn.LayerNorm(channels)

    def forward(self, charge, event, task):
        # BCHW -> BHWC for LayerNorm over channels, then restore BCHW.
        c = self.charge_norm(charge.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        e = self.event_norm(event.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        w = self.alpha[task].sigmoid().view(1, -1, 1, 1)
        return w * c + (1.0 - w) * e


def run_checks(out_dir: Path):
    torch.manual_seed(0)
    tasks = ("semseg", "depth", "normals", "edge")
    c = 64
    film, lora = ZeroFiLM(c), ZeroLoRA(c, rank=8)
    calib, gate = DecoupledCalibration(c, tasks), DualPathGate(c, tasks)
    x = torch.randn(2, c, 7, 9)
    base = x.clone()
    step0 = (film(x) - base).abs().max().item()
    step0 = max(step0, (lora(x) - base).abs().max().item())
    # Calibration is identity at construction; the dual gate is tested for
    # finite output and prior weights rather than identity.
    cal0 = max((calib(x, t) - x).abs().max().item() for t in tasks)
    losses = []
    for t in tasks:
        y = gate(lora(film(x)), x, t)
        losses.append(calib(y, t).square().mean())
    total = sum(losses)
    total.backward()
    grad = {"film": float(film.gamma.grad.abs().sum() + film.beta.grad.abs().sum()),
            "lora_A": float(lora.lora_A.weight.grad.abs().sum()),
            "lora_B": float(lora.lora_B.weight.grad.abs().sum()),
            "calibration": float(sum(p.grad.abs().sum() for p in calib.parameters())),
            "gate": float(sum(p.grad.abs().sum() for p in gate.parameters()))}
    trainable = sum(p.numel() for m in (film, lora, calib, gate) for p in m.parameters() if p.requires_grad)
    result = {
        "step0_film_lora_max_abs": step0,
        "step0_calibration_max_abs": cal0,
        "zero_lora_B_max_abs": float(lora.lora_B.weight.abs().max()),
        "trainable_parameters": trainable,
        "gradient_sums": grad,
        "prior_weights": {t: float(gate.alpha[t].sigmoid().mean()) for t in tasks},
        # At strict zero-init, lora_A has zero first-step gradient by design:
        # lora_B is the active gate and must receive the gradient first.
        "passed": step0 == 0.0 and cal0 == 0.0 and float(lora.lora_B.weight.abs().max()) == 0.0 and trainable <= 300000 and grad["film"] > 0 and grad["lora_B"] > 0 and grad["calibration"] > 0 and grad["gate"] > 0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase2_invariants.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Phase 2 invariant failed")


if __name__ == "__main__":
    run_checks(Path(__file__).parent / "phase2_results")
