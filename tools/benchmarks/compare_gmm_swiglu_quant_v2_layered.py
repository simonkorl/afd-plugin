"""Test A3 A8W4 per-channel quantization against the layered variant.

Use --original-only for an original-operator smoke test, not an accuracy test.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import numpy as np

INT4_PER_INT32 = 8
NZ_K_BLOCK = 16
NZ_INT4_N_BLOCK = 64
INT4_MIN = -8
INT4_MAX = 8
DEFAULT_COUNTS = (17, 0, 63, 48)
INT8_MIN = -128
INT8_MAX = 128
MSD_WEIGHT_CORRECTION = 8.0
DIMENSION_ALIGNMENT = 256
A8W4_MAX_COLUMNS = 10240
A8W4_HIDDEN_LIMIT = 20000


def _pack_int4(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] % INT4_PER_INT32:
        raise ValueError("the final dimension must be divisible by 8")
    grouped = values.reshape(*values.shape[:-1], -1, INT4_PER_INT32)
    shifts = np.arange(INT4_PER_INT32, dtype=np.uint32) * 4
    packed = np.bitwise_or.reduce(
        (grouped.astype(np.uint32) & 0xF) << shifts, axis=-1
    )
    return np.ascontiguousarray(packed.view(np.int32))


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


def _invoke_original(op, inputs, group_list_type: int):
    x, weight, weight_scale, weight_assist, x_scale, group_list = inputs
    nz_weights = []
    for packed_weight in weight:
        experts, hidden, packed_columns = packed_weight.shape
        nz_weights.append(
            packed_weight.reshape(
                experts,
                hidden // NZ_K_BLOCK,
                NZ_K_BLOCK,
                packed_columns // (NZ_INT4_N_BLOCK // INT4_PER_INT32),
                NZ_INT4_N_BLOCK // INT4_PER_INT32,
            ).permute(0, 3, 1, 2, 4).contiguous()
        )
    return op(
        x,
        nz_weights,
        weight_scale,
        x_scale,
        group_list,
        smooth_scale=None,
        weight_assist_matrix=weight_assist,
        bias=None,
        dequant_mode=0,
        quant_mode=0,
        group_list_type=group_list_type,
        tuning_config=None,
    )


def _invoke_layered(torch, inputs, layer: int, group_list_type: int):
    x, weights, scales, weight_assist, x_scale, group_list = inputs
    return torch.ops.afd_ascend.gmm_swiglu_quant_v2_layered(
        x,
        weights,
        scales,
        weight_assist,
        x_scale,
        group_list,
        torch.tensor([layer], device=x.device, dtype=torch.int64),
        0,
        0,
        group_list_type,
        None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-op", default="npu.npu_grouped_matmul_swiglu_quant_v2")
    parser.add_argument(
        "--original-only", action="store_true",
        help="skip layered loading/comparison; validate output shape, dtype and finite scales",
    )
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
    if args.experts < 1 or len(args.tokens) != args.experts:
        parser.error("--tokens must contain exactly --experts values")
    if (
        args.layers < 1 or args.hidden <= 0 or args.columns <= 0
        or args.hidden % DIMENSION_ALIGNMENT or args.columns % DIMENSION_ALIGNMENT
    ):
        parser.error("layers must be positive and hidden/columns must be multiples of 256")
    if args.hidden >= A8W4_HIDDEN_LIMIT or args.columns > A8W4_MAX_COLUMNS:
        parser.error("A3 A8W4 requires hidden < 20000 and columns <= 10240")
    if min(args.tokens) < 0 or sum(args.tokens) == 0:
        parser.error("token counts must be nonnegative with a positive total")
    for value in (args.atol, args.rtol, args.dequant_atol, args.dequant_rtol):
        if not np.isfinite(value) or value < 0:
            parser.error("tolerances must be finite and nonnegative")

    torch = importlib.import_module("torch")
    importlib.import_module("torch_npu")
    if not args.original_only:
        from afd_plugin.compat.npu.ops import ensure_afd_ascend_ops_loaded

        ensure_afd_ascend_ops_loaded()
    device = torch.device(args.device)
    if device.type != "npu":
        parser.error("--device must be npu or npu:<index> on an Atlas A3")
    device_index = (
        device.index if device.index is not None else int(os.environ.get("LOCAL_RANK", "0"))
    )
    torch.npu.set_device(device_index)
    original = _resolve_original(torch, args.original_op)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    counts = torch.tensor(args.tokens, device=args.device, dtype=torch.int64)
    total_tokens = int(sum(args.tokens))
    x = torch.randint(
        INT8_MIN,
        INT8_MAX,
        (total_tokens, args.hidden),
        generator=generator,
        device="cpu",
        dtype=torch.int8,
    ).to(args.device)
    weights = []
    scales = []
    weight_assists = []
    for _ in range(args.layers):
        weight = torch.randint(
            INT4_MIN,
            INT4_MAX,
            (args.experts, args.hidden, args.columns),
            generator=generator,
            device="cpu",
            dtype=torch.int32,
        )
        weights.append(torch.from_numpy(_pack_int4(weight.numpy())).to(args.device))
        scale = torch.rand(
            (args.experts, args.columns), generator=generator, dtype=torch.float32
        ) * 0.04 + 0.01
        weight_assists.append(
            (MSD_WEIGHT_CORRECTION * weight.float().sum(dim=1) * scale).to(args.device)
        )
        scales.append(
            (scale.view(torch.int32).to(torch.int64) & 0xFFFFFFFF).to(args.device)
        )
    x_scale = (torch.rand(total_tokens, generator=generator) * 0.04 + 0.01).to(args.device)
    reports = []
    for group_list_type, group_list in ((0, counts.cumsum(0)), (1, counts)):
        for layer in range(args.layers):
            original_inputs = (
                x, [weights[layer]], [scales[layer]], [weight_assists[layer]],
                x_scale, group_list,
            )
            reference = _invoke_original(original, original_inputs, group_list_type)
            if args.original_only:
                torch.npu.synchronize()
                output, output_scale = reference
                report = {
                    "layer": layer,
                    "group_list_type": group_list_type,
                    "output_shape": list(output.shape),
                    "output_dtype": str(output.dtype),
                    "scale_shape": list(output_scale.shape),
                    "scale_dtype": str(output_scale.dtype),
                    "passed": (
                        tuple(output.shape) == (total_tokens, args.columns // 2)
                        and output.dtype == torch.int8
                        and tuple(output_scale.shape) == (total_tokens,)
                        and output_scale.dtype == torch.float32
                        and bool(torch.isfinite(output_scale).all().item())
                        and bool((output_scale >= 0).all().item())
                    ),
                }
                reports.append(report)
                print(json.dumps(report), flush=True)
                continue
            actual = _invoke_layered(
                torch, (x, weights, scales, weight_assists, x_scale, group_list),
                layer, group_list_type,
            )
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
            json.dumps(
                {"configuration": {**vars(args), "report": str(args.report)},
                 "cases": reports, "passed": passed}, indent=2,
            ),
            encoding="utf-8",
        )
    mode = "original-only smoke tests" if args.original_only else "comparisons"
    print(f"{'PASS' if passed else 'FAIL'}: {len(reports)} A3 A8W4 {mode}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
