"""Compare grouped_matmul_swiglu_quant_v2 with its layered variant."""

from __future__ import annotations

import torch
import argparse
import importlib
import json
import os
from pathlib import Path

import numpy as np

INT4_PER_INT32 = 8
INT4_MIN = -8
INT4_MAX = 8
DEFAULT_COUNTS = (17, 0, 63, 48)


def _pack_int4(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] % INT4_PER_INT32:
        raise ValueError("the final dimension must be divisible by 8")
    grouped = values.reshape(*values.shape[:-1], -1, INT4_PER_INT32)
    if isinstance(values, np.ndarray):
        shifts = np.arange(INT4_PER_INT32, dtype=np.uint32) * 4
        packed = np.bitwise_or.reduce(
            (grouped.astype(np.uint32) & 0xF) << shifts, axis=-1
        )
        return np.ascontiguousarray(packed.view(np.int32))
  
    packed = torch.zeros(
        grouped.shape[:-1], dtype=torch.int32, device=values.device
    )
    for index in range(INT4_PER_INT32):
        packed |= (grouped[..., index] & 0xF) << (index * 4)
    return packed.contiguous()


def _metrics(torch, actual, reference, atol: float, rtol: float) -> dict:
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        return {
            "passed": False,
            "reason": "shape or dtype mismatch",
            "actual_shape": list(actual.shape),
            "reference_shape": list(reference.shape),
            "actual_dtype": str(actual.dtype),
            "reference_dtype": str(reference.dtype),
        }
    actual_float = actual.float()
    reference_float = reference.float()
    error = (actual_float - reference_float).abs()
    valid = torch.isfinite(actual_float) & torch.isfinite(reference_float)
    valid &= error <= atol + rtol * reference_float.abs()
    return {
        "passed": bool(valid.all().item()),
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
        "exact_match_fraction": float((actual == reference).float().mean().item()),
        "failed_elements": int((~valid).sum().item()),
        "elements": actual.numel(),
    }


def _resolve_original(torch, name: str):
    namespace, operator = name.split(".", 1)
    return getattr(getattr(torch.ops, namespace), operator)


def _invoke_original(op, inputs):
    x, weight, weight_scale, x_scale, group_list = inputs
    return op(
        x,
        weight,
        weight_scale,
        None,
        None,
        x_scale,
        None,
        group_list,
        0,
        0,
        0,
        0,
        None,
    )


def _invoke_layered(torch, inputs, layer: int):
    x, weights, scales, x_scale, group_list = inputs
    return torch.ops.afd_ascend.gmm_swiglu_quant_v2_layered(
        x,
        weights,
        scales,
        [],
        x_scale,
        group_list,
        torch.tensor([layer], device=x.device, dtype=torch.int64),
        0,
        0,
        0,
        None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-op", default="npu.npu_grouped_matmul_swiglu_quant_v2")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--tokens", type=int, nargs="+", default=list(DEFAULT_COUNTS))
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--columns", type=int, default=512)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--dequant-atol", type=float, default=0.0)
    parser.add_argument("--dequant-rtol", type=float, default=0.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if len(args.tokens) != args.experts:
        parser.error("--tokens must contain exactly --experts values")
    if args.layers < 1 or args.hidden % 256 or args.columns % 256:
        parser.error("layers must be positive and hidden/columns must be multiples of 256")
    if min(args.tokens) < 0 or sum(args.tokens) == 0:
        parser.error("token counts must be nonnegative with a positive total")
    for value in (args.atol, args.rtol, args.dequant_atol, args.dequant_rtol):
        if not np.isfinite(value) or value < 0:
            parser.error("tolerances must be finite and nonnegative")

    torch = importlib.import_module("torch")
    importlib.import_module("torch_npu")
    from afd_plugin.compat.npu.ops import ensure_afd_ascend_ops_loaded

    ensure_afd_ascend_ops_loaded()
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    original = _resolve_original(torch, args.original_op)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    counts = torch.tensor(args.tokens, device=args.device, dtype=torch.int64)
    total_tokens = int(sum(args.tokens))
    x = torch.randint(
        INT4_MIN,
        INT4_MAX,
        (total_tokens, args.hidden),
        generator=generator,
        device="cpu",
        dtype=torch.int32,
    )
    x = _pack_int4(x.cpu()).to(args.device)
    weights = []
    scales = []
    for _ in range(args.layers):
        weight = torch.randint(
            INT4_MIN,
            INT4_MAX,
            (args.experts, args.hidden, args.columns),
            generator=generator,
            device="cpu",
            dtype=torch.int32,
        )
        weights.append(_pack_int4(weight).to(args.device))
        scale = torch.rand(
            (args.experts, args.columns), generator=generator, dtype=torch.float32
        ) * 0.04 + 0.01
        scales.append(
            (scale.view(torch.int32).to(torch.int64) & 0xFFFFFFFF).to(args.device)
        )
    x_scale = (torch.rand(total_tokens, generator=generator) * 0.04 + 0.01).to(args.device)
    reports = []
    for group_list_type, group_list in ((0, counts.cumsum(0)), (1, counts)):
        for layer in range(args.layers):
            original_inputs = (x, [weights[layer]], [scales[layer]], x_scale, group_list)
            reference = _invoke_original(original, original_inputs)
            actual = _invoke_layered(torch, (x, weights, scales, x_scale, group_list), layer)
            reference_y, reference_scale = reference
            actual_y, actual_scale = actual
            quantized = _metrics(torch, actual_y, reference_y, args.atol, args.rtol)
            scale = _metrics(torch, actual_scale, reference_scale, args.atol, args.rtol)
            dequantized = _metrics(
                torch,
                actual_y.float() * actual_scale.unsqueeze(-1),
                reference_y.float() * reference_scale.unsqueeze(-1),
                args.dequant_atol,
                args.dequant_rtol,
            )
            report = {
                "layer": layer,
                "group_list_type": group_list_type,
                "quantized": quantized,
                "scale": scale,
                "dequantized": dequantized,
                "passed": all(item["passed"] for item in (quantized, scale, dequantized)),
            }
            reports.append(report)
            print(json.dumps(report), flush=True)
    passed = all(item["passed"] for item in reports)
    if args.report:
        args.report.write_text(
            json.dumps({"configuration": vars(args), "cases": reports, "passed": passed}, indent=2),
            encoding="utf-8",
        )
    print(f"{'PASS' if passed else 'FAIL'}: {len(reports)} comparisons")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
