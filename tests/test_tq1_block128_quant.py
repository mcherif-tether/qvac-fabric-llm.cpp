#!/usr/bin/env python3
"""Integration test: synthetic + real GGUF -> llama-quantize --pure TQ1_0_128."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))

import gguf
from gguf.constants import GGMLQuantizationType, LlamaFileType

ROOT = Path(__file__).resolve().parent.parent

BLOCK_SIZE = 128
TYPE_SIZE = 28  # 24-byte qs + 2-byte qh + 2-byte fp16 d
SYNTHETIC_TENSOR = "blk.0.attn_q.weight"


# ---------------------------------------------------------------------------
# Model GGUF verification (real Qwen3 fixture outputs)
# ---------------------------------------------------------------------------


def tensor_data_bytes(reader: gguf.GGUFReader, index: int) -> int:
    tensors = reader.tensors
    start = tensors[index].data_offset
    if index + 1 < len(tensors):
        return tensors[index + 1].data_offset - start
    return len(reader.data) - start


def verify_tensor_values(
    r_out: gguf.GGUFReader,
    r_in: gguf.GGUFReader,
    tq1_tensors: list[gguf.ReaderTensor],
    q128: type,
) -> None:
    """Byte-exact parity: every TQ1_0_128 weight vs Python quant of the source F16/BF16/F32 tensor."""
    in_by_name = {t.name: t for t in r_in.tensors}
    float_types = {
        GGMLQuantizationType.F16,
        GGMLQuantizationType.BF16,
        GGMLQuantizationType.F32,
    }

    for t in tq1_tensors:
        tin = in_by_name.get(t.name)
        if tin is None:
            raise AssertionError(f"tensor {t.name!r} missing from input GGUF")
        if not np.array_equal(tin.shape, t.shape):
            raise AssertionError(
                f"tensor {t.name}: shape {tuple(t.shape)} != input {tuple(tin.shape)}"
            )
        if tin.tensor_type not in float_types:
            continue

        f32 = np.asarray(gguf.dequantize(tin.data, tin.tensor_type), dtype=np.float32)
        if f32.shape != tuple(t.shape):
            f32 = f32.reshape(tuple(t.shape))

        py_q = q128.quantize(f32)
        out_bytes = np.asarray(t.data, dtype=np.uint8).ravel()
        py_bytes = np.asarray(py_q, dtype=np.uint8).ravel()
        if not np.array_equal(py_bytes, out_bytes):
            diff_idx = int(np.argmax(py_bytes != out_bytes))
            raise AssertionError(
                f"tensor {t.name}: Python quant mismatch at byte {diff_idx} "
                f"(expected 0x{py_bytes[diff_idx]:02x}, got 0x{out_bytes[diff_idx]:02x})"
            )


def check_gguf(
    out_128: Path,
    out_256: Path | None,
    bf16_in: Path | None,
) -> None:
    block_size = BLOCK_SIZE
    type_size = TYPE_SIZE

    r128 = gguf.GGUFReader(str(out_128))
    tq1_tensors = [t for t in r128.tensors if t.tensor_type == GGMLQuantizationType.TQ1_0_128]
    if not tq1_tensors:
        raise AssertionError(f"no TQ1_0_128 tensors in {out_128}")

    for t in tq1_tensors:
        expected = t.n_elements // block_size * type_size
        actual = tensor_data_bytes(r128, r128.tensors.index(t))
        if actual != expected:
            raise AssertionError(
                f"tensor {t.name}: payload={actual} bytes, expected {expected} "
                f"(ne={t.n_elements}, block_size={block_size})"
            )

    t0 = tq1_tensors[0]
    t0_idx = r128.tensors.index(t0)
    raw = r128.data[t0.data_offset : t0.data_offset + tensor_data_bytes(r128, t0_idx)]
    if len(raw) < type_size or len(raw) % type_size != 0:
        raise AssertionError("tensor data not aligned to block size")

    q128 = gguf.quants.TQ1_0_128
    row_bytes = (t0.shape[-1] // block_size) * type_size if len(t0.shape) >= 1 else type_size
    dq = q128.dequantize(raw[:row_bytes].copy().reshape(1, -1))
    if not np.isfinite(dq).all():
        raise AssertionError("dequantized values contain non-finite numbers")

    if out_256 is not None:
        r256 = gguf.GGUFReader(str(out_256))
        t256_list = [t for t in r256.tensors if t.tensor_type == GGMLQuantizationType.TQ1_0]
        if not t256_list:
            raise AssertionError(f"no TQ1_0 tensors in {out_256}")
        if out_128.stat().st_size == out_256.stat().st_size:
            raise AssertionError(
                "block-128 and block-256 outputs have identical file size "
                "(expected block-128 slightly larger)"
            )

        t256 = next(t for t in r256.tensors if t.name == t0.name)
        idx256 = r256.tensors.index(t256)
        raw256 = r256.data[t256.data_offset : t256.data_offset + tensor_data_bytes(r256, idx256)]
        raw128 = r128.data[t0.data_offset : t0.data_offset + tensor_data_bytes(r128, t0_idx)]
        if np.array_equal(raw128, raw256):
            raise AssertionError(f"tensor {t0.name} bytes identical between block 128 and 256")

    if bf16_in is not None:
        rin = gguf.GGUFReader(str(bf16_in))
        verify_tensor_values(r128, rin, tq1_tensors, q128)


# ---------------------------------------------------------------------------
# Synthetic GGUF verification (minimal llama-arch fixture)
# ---------------------------------------------------------------------------


def make_alternating_weights(rows: int, cols: int) -> np.ndarray:
    if cols % BLOCK_SIZE != 0:
        raise ValueError(f"cols must be a multiple of {BLOCK_SIZE}")
    data = np.zeros((rows, cols), dtype=np.float16)
    n_blocks = cols // BLOCK_SIZE
    for b in range(n_blocks):
        value = np.float16(1.0 if b % 2 == 0 else 0.0)
        data[:, b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE] = value
    return data


def make_random_weights(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-3.0, 3.0, size=(rows, cols)).astype(np.float16)


def make_ramp_weights(rows: int, cols: int) -> np.ndarray:
    x = np.linspace(-1.5, 1.5, cols, dtype=np.float32)
    return np.tile(x, (rows, 1)).astype(np.float16)


def make_ternary_weights(rows: int, cols: int) -> np.ndarray:
    vals = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            vals[i, j] = float((i + j) % 3 - 1)
    return vals.astype(np.float16)


def write_synthetic_f16_gguf(path: Path, data: np.ndarray) -> None:
    writer = gguf.GGUFWriter(str(path), arch="llama")
    writer.add_file_type(int(LlamaFileType.MOSTLY_F16))
    writer.add_uint32("llama.block_count", 1)
    writer.add_uint32("llama.embedding_length", int(data.shape[0]))
    writer.add_uint32("llama.attention.head_count", 4)
    writer.add_uint32("llama.attention.head_count_kv", 4)
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-5)
    writer.add_uint32("llama.feed_forward_length", int(data.shape[1]))
    writer.add_uint32("llama.context_length", 512)
    writer.add_tensor(SYNTHETIC_TENSOR, data, raw_dtype=GGMLQuantizationType.F16)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def read_synthetic_quantized_bytes(path: Path, rows: int, cols: int) -> np.ndarray:
    reader = gguf.GGUFReader(str(path))
    tensor = next((t for t in reader.tensors if t.name == SYNTHETIC_TENSOR), None)
    if tensor is None:
        raise AssertionError(f"missing tensor {SYNTHETIC_TENSOR!r} in {path}")
    if tensor.tensor_type != GGMLQuantizationType.TQ1_0_128:
        raise AssertionError(f"{SYNTHETIC_TENSOR} type {tensor.tensor_type}, expected TQ1_0_128")

    n_blocks = cols // BLOCK_SIZE
    expected_bytes = rows * n_blocks * TYPE_SIZE
    raw = np.asarray(tensor.data, dtype=np.uint8).ravel()
    if raw.size != expected_bytes:
        raise AssertionError(f"payload {raw.size} bytes, expected {expected_bytes}")
    return raw


def verify_alternating_blocks(path: Path, rows: int, cols: int) -> None:
    raw = read_synthetic_quantized_bytes(path, rows, cols)
    n_blocks = cols // BLOCK_SIZE

    row0 = raw[: n_blocks * TYPE_SIZE]
    for b in range(n_blocks):
        block = row0[b * TYPE_SIZE : (b + 1) * TYPE_SIZE]
        qs, qh, d_bytes = block[:24], block[24:26], block[26:28]
        plus = b % 2 == 0
        expected_qs = 0xFF if plus else 0x80
        expected_qh = 0xFD if plus else 0x7F
        if not np.all(qs == expected_qs):
            raise AssertionError(
                f"block {b}: qs mismatch (got 0x{qs[0]:02x}.., expected 0x{expected_qs:02x})"
            )
        if not np.all(qh == expected_qh):
            raise AssertionError(
                f"block {b}: qh mismatch (got 0x{qh[0]:02x}, expected 0x{expected_qh:02x})"
            )
        d = d_bytes.view(np.float16)[0]
        expected_d = np.float16(1.0 if b % 2 == 0 else 0.0)
        if d != expected_d:
            raise AssertionError(f"block {b}: scale {d} != {expected_d}")

    f32 = gguf.quants.TQ1_0_128.dequantize(raw.reshape(1, -1)).reshape(rows, cols)
    for b in range(n_blocks):
        sl = slice(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE)
        expected = 1.0 if b % 2 == 0 else 0.0
        if not np.allclose(f32[:, sl], expected, atol=1e-6):
            raise AssertionError(f"dequant block {b} not all {expected}")


def verify_reference_parity(path: Path, source_f16: np.ndarray) -> None:
    rows, cols = source_f16.shape
    raw = read_synthetic_quantized_bytes(path, rows, cols)
    f32 = source_f16.astype(np.float32)
    expected = np.asarray(gguf.quants.TQ1_0_128.quantize(f32), dtype=np.uint8).ravel()
    if not np.array_equal(expected, raw):
        diff = int(np.argmax(expected != raw))
        raise AssertionError(
            f"byte mismatch at offset {diff} "
            f"(expected 0x{expected[diff]:02x}, got 0x{raw[diff]:02x})"
        )

    dq = gguf.quants.TQ1_0_128.dequantize(raw.reshape(1, -1)).reshape(rows, cols)
    ref_dq = gguf.quants.TQ1_0_128.dequantize(expected.reshape(1, -1)).reshape(rows, cols)
    if not np.isfinite(dq).all():
        raise AssertionError("dequantized values are not finite")
    if not np.allclose(dq, ref_dq, atol=1e-6):
        raise AssertionError("dequant does not match reference")


def run_synthetic_case(
    work: Path,
    quantize: Path,
    name: str,
    data: np.ndarray,
    verify: Callable[[Path, np.ndarray, int, int], None],
) -> None:
    rows, cols = data.shape
    f16_in = work / f"{name}-f16.gguf"
    out = work / f"{name}-tq1-128.gguf"

    write_synthetic_f16_gguf(f16_in, data)
    subprocess.run(
        [str(quantize), "--pure", str(f16_in), str(out), "TQ1_0_128"],
        check=True,
    )
    verify(out, data, rows, cols)


def verify_alternating_case(path: Path, _data: np.ndarray, rows: int, cols: int) -> None:
    verify_alternating_blocks(path, rows, cols)


def verify_reference_case(path: Path, data: np.ndarray, _rows: int, _cols: int) -> None:
    verify_reference_parity(path, data)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str | None = None


class TestRunner:
    def __init__(self) -> None:
        self.results: list[TestResult] = []

    def run(self, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            detail = str(exc) or type(exc).__name__
            self.results.append(TestResult(name, False, detail))
            print(f"FAIL: {name} — {detail}")
            return
        self.results.append(TestResult(name, True))
        print(f"PASS: {name}")

    def summary(self) -> tuple[int, int]:
        passed = sum(1 for r in self.results if r.passed)
        return passed, len(self.results)


def ensure_fixture(build_dir: Path, bf16: Path | None) -> Path:
    if bf16 is not None:
        if not bf16.is_file():
            raise FileNotFoundError(f"input GGUF not found: {bf16}")
        return bf16

    fixture_dir = build_dir / "test-fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    path = fixture_dir / "Qwen3-0.6B-f16.gguf"
    if path.is_file():
        return path

    hf = ROOT / "scripts" / "hf.sh"
    if not hf.is_file():
        raise FileNotFoundError(
            f"fixture missing at {path} and {hf} not found to download it"
        )

    print(f"Downloading tiny f16 fixture to {path} ...")
    subprocess.run(
        [str(hf), "--repo", "ggml-org/Qwen3-0.6B-GGUF", "--file", "Qwen3-0.6B-f16.gguf"],
        cwd=fixture_dir,
        check=True,
    )
    if not path.is_file():
        raise FileNotFoundError(f"download did not produce {path}")
    return path


def run_quantize(
    quantize: Path,
    inp: Path,
    out: Path,
    ftype: str,
    *,
    block_size: int | None = None,
) -> None:
    cmd = [str(quantize), "--pure"]
    if block_size is not None:
        cmd.extend(["--block-size", str(block_size)])
    cmd.extend([str(inp), str(out), ftype])
    subprocess.run(cmd, check=True)


def test_model_load_weights(build_dir: Path, model_path: Path) -> None:
    loader = build_dir / "bin" / "test-model-load-memory"
    if not loader.is_file():
        raise FileNotFoundError(
            f"{loader} not found — build with: cmake --build {build_dir.name} -j --target test-model-load-memory"
        )
    proc = subprocess.run([str(loader), str(model_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise AssertionError(detail or f"model load failed with exit code {proc.returncode}")
    if "Failed to load model" in (proc.stdout + proc.stderr):
        raise AssertionError("model loader reported failure")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("build_dir", nargs="?", default="build", help="cmake build directory")
    p.add_argument("bf16_in", nargs="?", default=None, help="optional F16/BF16 input GGUF")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for synthetic random cases")
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--cols", type=int, default=512)
    args = p.parse_args()

    build_dir = (ROOT / args.build_dir).resolve()
    quantize = build_dir / "bin" / "llama-quantize"
    if not quantize.is_file():
        print(
            f"error: {quantize} not found — build with: "
            f"cmake -B {args.build_dir} && cmake --build {args.build_dir} -j --target llama-quantize",
            file=sys.stderr,
        )
        return 1

    if args.cols % BLOCK_SIZE != 0:
        print(f"error: --cols must be a multiple of {BLOCK_SIZE}", file=sys.stderr)
        return 1

    bf16_in = Path(args.bf16_in).resolve() if args.bf16_in else None

    work = Path(tempfile.mkdtemp(prefix="tq1-bs128-"))
    syn_work = work / "synthetic"
    syn_work.mkdir()
    runner = TestRunner()

    out_128 = work / "tq1-128.gguf"
    out_256 = work / "tq1-256.gguf"
    out_128_alias = work / "tq1-128-alias.gguf"
    bad_out = work / "bad.gguf"

    try:
        bf16 = ensure_fixture(build_dir, bf16_in)

        synthetic_cases: list[tuple[str, np.ndarray, Callable]] = [
            ("alternating-1-0", make_alternating_weights(args.rows, args.cols), verify_alternating_case),
            ("random-uniform", make_random_weights(args.rows, args.cols, args.seed), verify_reference_case),
            ("linear-ramp", make_ramp_weights(args.rows, args.cols), verify_reference_case),
            ("ternary-pattern", make_ternary_weights(args.rows, args.cols), verify_reference_case),
            ("random-small", make_random_weights(64, 256, args.seed + 1), verify_reference_case),
        ]

        for case_name, data, verify in synthetic_cases:
            def synthetic_test(
                n: str = case_name,
                d: np.ndarray = data,
                v: Callable = verify,
            ) -> None:
                run_synthetic_case(syn_work, quantize, n, d, v)

            runner.run(f"synthetic/{case_name}", synthetic_test)

        runner.run(
            "model/quantize-tq1-0-128",
            lambda: run_quantize(quantize, bf16, out_128, "TQ1_0_128"),
        )
        runner.run(
            "model/quantize-block-size-128-alias",
            lambda: run_quantize(quantize, bf16, out_128_alias, "TQ1_0", block_size=128),
        )
        runner.run(
            "model/quantize-tq1-0-256-regression",
            lambda: run_quantize(quantize, bf16, out_256, "TQ1_0"),
        )

        def reject_invalid_block_size() -> None:
            proc = subprocess.run(
                [
                    str(quantize),
                    "--pure",
                    "--block-size",
                    "64",
                    str(bf16),
                    str(bad_out),
                    "TQ1_0",
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                raise AssertionError("--block-size 64 should fail")

        runner.run("model/reject-invalid-block-size-64", reject_invalid_block_size)

        runner.run(
            "model/gguf-parity-tq1-128-and-regression",
            lambda: check_gguf(out_128, out_256, bf16),
        )
        runner.run(
            "model/gguf-parity-block-size-alias",
            lambda: check_gguf(out_128_alias, None, bf16),
        )
        runner.run(
            "model/load-tq1-128-weights",
            lambda: test_model_load_weights(build_dir, out_128),
        )

    except Exception as exc:
        print(f"error: setup failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)

    passed, total = runner.summary()
    print(f"\n({passed}/{total}) tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
