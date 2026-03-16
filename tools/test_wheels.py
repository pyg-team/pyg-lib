"""Modal app for testing pre-built pyg-lib wheels on real GPUs.

Usage:
    # Run tests for a specific combination:
    modal run tools/test_wheels.py \
        --torch-version 2.10.0 --cuda-version cu130 --python-version 3.12

    # Run on a specific GPU:
    modal run tools/test_wheels.py \
        --torch-version 2.10.0 --cuda-version cu130 --python-version 3.12 \
        --gpu T4

    # Force rebuild images (bust Modal cache to pick up latest nightly):
    modal run tools/test_wheels.py \
        --torch-version 2.10.0 --cuda-version cu130 --python-version 3.12 \
        --force-build
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("pyg-lib-wheel-tests")

# https://modal.com/pricing
GPU_TYPES = [
    # Blackwell (sm_100)
    "B200",
    # Hopper (sm_90)
    "H100",
    # "H200",
    # Ada Lovelace (sm_89)
    "L4",
    # "L40S"
    # "RTX-PRO-6000",
    # Ampere (sm_86, sm_80)
    "A10",
    # "A100-40GB",
    # "A100-80GB",
    # Turing (sm_75)
    "T4",
]
MATRIX = [
    {
        "torch": "2.11.0",
        "python": ["3.10", "3.11", "3.12", "3.13", "3.14"],
        "cuda": ["cu126", "cu128", "cu130"],
    },
    {
        "torch": "2.10.0",
        "python": ["3.10", "3.11", "3.12", "3.13", "3.14"],
        "cuda": ["cu126", "cu128", "cu130"],
    },
    {
        "torch": "2.9.0",
        "python": ["3.10", "3.11", "3.12", "3.13"],
        "cuda": ["cu126", "cu128", "cu130"],
    },
    {
        "torch": "2.8.0",
        "python": ["3.10", "3.11", "3.12", "3.13"],
        "cuda": ["cu126", "cu128", "cu129"],
    },
]

repo_root = Path(__file__).resolve().parent.parent


def _filter_combos(
    torch_version: str,
    cuda_version: str,
    python_version: str,
    gpu: str,
) -> list[tuple[str, str, str, str]]:
    """Return (torch, python, cuda, gpu) tuples matching the given filters."""
    combos = []
    for entry in MATRIX:
        if torch_version and entry["torch"] != torch_version:
            continue
        for pv in entry["python"]:
            if python_version and pv != python_version:
                continue
            for cv in entry["cuda"]:
                if cuda_version and cv != cuda_version:
                    continue
                for g in GPU_TYPES:
                    if gpu and g != gpu:
                        continue
                    combos.append((entry["torch"], pv, cv, g))
    return combos


def _make_image(
    torch_version: str,
    cuda_short: str,
    python_version: str,
    force_build: bool = False,
) -> modal.Image:
    torch_index = f"https://download.pytorch.org/whl/{cuda_short}"
    base = "https://data.pyg.org/whl"
    suffix = f"torch-{torch_version}+{cuda_short}.html"
    nightly_url = f"{base}/nightly/{suffix}"
    stable_url = f"{base}/{suffix}"
    return (modal.Image.debian_slim(python_version=python_version).pip_install(
        f"torch=={torch_version}",
        index_url=torch_index,
    ).pip_install(
        "pytest",
        "scipy",
    ).pip_install(
        "pyg-lib",
        find_links=nightly_url,
        force_build=force_build,
    ).pip_install(
        "torch-spline-conv",
        find_links=stable_url,
    ).run_commands(
        "python -c \"import pyg_lib; print('pyg-lib:', pyg_lib.__version__)\"",
    ).add_local_file(
        repo_root / "pyproject.toml",
        remote_path="/root/pyproject.toml",
    ).add_local_dir(repo_root / "test", remote_path="/root/test"))


def _build_images(
    combos: list[tuple[str, str, str, str]],
    force_build: bool,
) -> dict[tuple[str, str, str], modal.Image]:
    """Build one image per unique (torch, python, cuda) triple."""
    images: dict[tuple[str, str, str], modal.Image] = {}
    for tv, pv, cv, _ in combos:
        key = (tv, pv, cv)
        if key not in images:
            images[key] = _make_image(tv, cv, python_version=pv,
                                      force_build=force_build)
    return images


def _run_sandboxes(
    combos: list[tuple[str, str, str, str]],
    images: dict[tuple[str, str, str], modal.Image],
) -> list[tuple[str, int, str, str]]:
    """Launch all sandboxes in parallel, wait, and return results."""
    sandboxes = []
    for tv, pv, cv, g in combos:
        label = f"torch {tv} + {cv} + py{pv} on {g}"
        sb = modal.Sandbox.create(
            "python",
            "-m",
            "pytest",
            "--rootdir=/root",
            "/root/test",
            app=app,
            image=images[(tv, pv, cv)],
            gpu=g,
            workdir="/root",
            timeout=30,
        )
        sandboxes.append((label, sb))

    results = []
    for label, sb in sandboxes:
        sb.wait()
        results.append((
            label,
            sb.returncode,
            sb.stdout.read(),
            sb.stderr.read(),
        ))
    return results


def _print_results(results: list[tuple[str, int, str, str]]) -> int:
    """Print summary table and full logs. Return number of failures."""
    passed = sum(1 for _, rc, _, _ in results if rc == 0)
    failed = len(results) - passed

    print()
    print(f"{'Combination':<45} {'Result'}")
    print("-" * 55)
    for label, rc, _, _ in results:
        print(f"{label:<45} {'PASS' if rc == 0 else 'FAIL'}")
    print()
    print(f"Total: {passed} passed, {failed} failed out of {len(results)}")

    for label, rc, stdout, stderr in results:
        print()
        print(f"=== {label} ===")
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)

    return failed


@app.local_entrypoint()
def main(
    torch_version: str,
    cuda_version: str,
    python_version: str,
    gpu: str = "",
    force_build: bool = False,
):
    """Run wheel tests, optionally filtered by torch/cuda/gpu."""
    combos = _filter_combos(torch_version, cuda_version, python_version, gpu)
    if not combos:
        print("No matching combinations found.")
        return

    print(f"Running {len(combos)} test combo(s)...")
    images = _build_images(combos, force_build)
    results = _run_sandboxes(combos, images)
    if _print_results(results):
        raise SystemExit(1)
