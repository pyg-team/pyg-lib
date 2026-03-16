"""Modal app for testing pre-built pyg-lib wheels on real GPUs.

Usage:
    # Run full matrix (all torch/cuda combos on T4):
    modal run tools/test_wheels.py

    # Run specific combination:
    modal run tools/test_wheels.py --torch-version 2.10.0 --cuda-version cu130

    # Run on a specific GPU:
    modal run tools/test_wheels.py \
        --torch-version 2.10.0 --cuda-version cu128 --gpu T4
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("pyg-lib-wheel-tests")

# Matches the supported CUDA wheel matrix from README.md.
MATRIX = [
    {
        "torch": "2.10.0",
        "cuda": "cu126"
    },
    {
        "torch": "2.10.0",
        "cuda": "cu128"
    },
    {
        "torch": "2.10.0",
        "cuda": "cu130"
    },
    {
        "torch": "2.9.0",
        "cuda": "cu126"
    },
    {
        "torch": "2.9.0",
        "cuda": "cu128"
    },
    {
        "torch": "2.9.0",
        "cuda": "cu130"
    },
    {
        "torch": "2.8.0",
        "cuda": "cu126"
    },
    {
        "torch": "2.8.0",
        "cuda": "cu128"
    },
    {
        "torch": "2.8.0",
        "cuda": "cu129"
    },
]

GPU_TYPES = ["T4"]  # Extend: "L4", "A100-80GB", "H100", "B200"

CUDA_VERSION_MAP = {
    "cu126": "12.6",
    "cu128": "12.8",
    "cu129": "12.9",
    "cu130": "13.0",
}

repo_root = Path(__file__).resolve().parent.parent


def _make_image(torch_version: str, cuda_short: str) -> modal.Image:
    cuda_full = CUDA_VERSION_MAP[cuda_short]
    tag = f"{torch_version}-cuda{cuda_full}-cudnn9-runtime"
    base = "https://data.pyg.org/whl"
    suffix = f"torch-{torch_version}+{cuda_short}.html"
    nightly_url = f"{base}/nightly/{suffix}"
    stable_url = f"{base}/{suffix}"
    return (modal.Image.from_registry(
        f"pytorch/pytorch:{tag}",
        setup_dockerfile_commands=[
            # Remove PEP 668 marker so Modal's pip bootstrap works.
            "RUN rm -f /usr/lib/python*/EXTERNALLY-MANAGED",
        ],
    ).pip_install("pyg-lib", find_links=nightly_url).pip_install(
        "torch-spline-conv", find_links=stable_url).pip_install(
            "pytest", "scipy").add_local_dir(repo_root / "test",
                                             remote_path="/root/test"))


@app.local_entrypoint()
def main(
    torch_version: str = "",
    cuda_version: str = "",
    gpu: str = "",
):
    """Run wheel tests, optionally filtered by torch/cuda/gpu."""
    combos = []
    for entry in MATRIX:
        if torch_version and entry["torch"] != torch_version:
            continue
        if cuda_version and entry["cuda"] != cuda_version:
            continue
        for g in GPU_TYPES:
            if gpu and g != gpu:
                continue
            combos.append((entry["torch"], entry["cuda"], g))

    if not combos:
        print("No matching combinations found.")
        return

    print(f"Running {len(combos)} test combo(s)...")

    # Launch all sandboxes in parallel.
    sandboxes = []
    for tv, cv, g in combos:
        label = f"torch {tv} + {cv} on {g}"
        image = _make_image(tv, cv)
        sb = modal.Sandbox.create(
            "python",
            "-m",
            "pytest",
            "-x",
            "-v",
            "/root/test",
            app=app,
            image=image,
            gpu=g,
            timeout=600,
        )
        sandboxes.append((label, sb))

    # Wait for all sandboxes and collect results.
    passed = 0
    failed = 0
    print()
    print(f"{'Combination':<45} {'Result'}")
    print("-" * 55)
    for label, sb in sandboxes:
        sb.wait()
        exit_code = sb.returncode
        status = "PASS" if exit_code == 0 else "FAIL"
        print(f"{label:<45} {status}")
        if exit_code == 0:
            passed += 1
        else:
            failed += 1
            stdout = sb.stdout.read()
            stderr = sb.stderr.read()
            print(stdout[-2000:] if len(stdout) > 2000 else stdout)
            print(stderr[-2000:] if len(stderr) > 2000 else stderr)

    print()
    print(f"Total: {passed} passed, {failed} failed out of {passed + failed}")
    if failed:
        raise SystemExit(1)
