---
name: support-new-pytorch
description: Add or update PyTorch version support in pyg-lib CI, wheels, and docs. Use when a new PyTorch version reaches any release stage — nightly, RC (test index), final release, or patch release (e.g. "support PyTorch 2.13", "2.13 RC is out", "add the 2.13.1 index") — or when dropping support for an old PyTorch version.
---

# Support a new PyTorch version

pyg-lib builds C++/CUDA wheels against specific PyTorch versions. Supporting a
new version is a staged process that follows PyTorch's own release lifecycle.
Each stage below is one small PR (see the example PRs). Never do all stages at
once — the files change depending on where PyTorch is in its release cycle.

## Step 0: Determine the release stage

Ask (or check https://pypi.org/project/torch/#history and
https://github.com/pytorch/pytorch/releases) which stage PyTorch `X.Y` is at:

| Stage | Signal | Go to |
|-------|--------|-------|
| Nightly only | `X.Y.0.devYYYYMMDD` on https://download.pytorch.org/whl/nightly | Stage 1 |
| RC | `X.Y.0` available on https://download.pytorch.org/whl/test | Stage 2 |
| Final | `X.Y.0` on PyPI / https://download.pytorch.org/whl | Stage 3 |
| Patch | `X.Y.1` (or `.2`) released, wheels are ABI-compatible | Stage 4 |
| — | A new stable exists, oldest version can go (keep ~3) | Stage 5 |

Also check whether the new PyTorch adds a **new CUDA version** (compare the
`cuda-version` lists in `.github/workflows/utils/*.json` against the CUDA
variants PyTorch publishes). If yes, do "New CUDA version" (below) first.

## Key convention: the rolling pre-release placeholder

The **highest** torch version referenced in CI is a placeholder for
"pre-release" and is special-cased in exactly three install sites:

1. `.github/actions/setup/action.yml` — composite setup action (used by
   `_build_windows.yml`, `cpp_testing.yml`, `documentation.yml`). Also holds
   the default `torch-version` (= current stable).
2. `.github/workflows/utils/build_linux.sh` — sets `CIBW_BEFORE_BUILD` /
   `CIBW_BEFORE_TEST` for cibuildwheel.
3. `.github/workflows/utils/build_macos.sh` — same for macOS.

The `if` branch in each installs from a different index per stage:

- nightly: `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/${CUDA}`
- RC:      `pip install torch==X.Y.0 --index-url https://download.pytorch.org/whl/test/${CUDA}`
- stable:  `pip install torch==X.Y.0 --index-url https://download.pytorch.org/whl/${CUDA}` (the `else` branch)

When `X.Y` goes final, the placeholder version in the `if` branches moves
forward to `X.(Y+1).0`.

## Build matrices

`.github/workflows/utils/{full,minimal}_matrix_{linux,macos,windows}.json`

- `full_*`: used by nightly/final wheel releases (`release.yml`, daily cron at
  4:00 UTC) and by PRs labeled `ci: full`.
- `minimal_*`: default for PR CI. Keep it small (2 entries: newest pre-release
  config + one older stable config; 2 Python versions).
- Linux entries also carry `"arch": ["x86_64", "aarch64"]`.
- PR builds only run when the PR has `os: linux` / `os: macos` / `os: windows`
  labels (`pull.yml` reads labels) — remind the author to add them.

## Stage 1: Build against PyTorch X.Y nightlies

Usually started to get early signal and/or to ship wheels for a new CUDA
version. Prerequisite: the placeholder `if` branches already point at `X.Y.0`
(done when `X.(Y-1)` went final, Stage 3).

1. Add an `X.Y.0` entry to `full_matrix_linux.json` and
   `minimal_matrix_linux.json`. Start narrow — newest CUDA + newest Python
   only (example: PR #611 added `2.12.0` with `cu132`/`3.14` only) — and
   widen in follow-ups once green (PR #629).
2. Windows support is a follow-up PR (PR #630): add the matrix entry to
   `*_matrix_windows.json`; if a new CUDA is involved, extend
   `.github/workflows/cuda/Windows.sh` and `Windows-env.sh` first.
3. macOS: add `X.Y.0` to `full_matrix_macos.json` / bump
   `minimal_matrix_macos.json` (PR #637).
4. `README.md`: add a "PyTorch Nightly" support table (columns = CUDA
   versions, rows = Linux/Windows/macOS) above the stable tables.
5. `CHANGELOG.md`: entry under `## [Unreleased]` → `### Added`, e.g.
   `- Added support for PyTorch \`2.12.0+cu132\` wheels in nightly releases ([#615](https://github.com/pyg-team/pyg-lib/pull/615))`.
   (CI enforces a CHANGELOG entry unless the PR has the `skip-changelog` label.)

Nightly pyg-lib wheels then appear under
https://data.pyg.org/whl/nightly/torch-X.Y.0+${CUDA}.html after the next
nightly `release.yml` run.

## Stage 2: PyTorch X.Y RC (test index)

Example: PR #641. In all three install sites:

1. Move the nightly `if` branch from `X.Y.0` to `X.(Y+1).0` (the placeholder
   rolls forward; next nightlies are versioned `X.(Y+1).0.dev*`).
2. Add an `elif` for `X.Y.0` installing from the test index:
   `torch==X.Y.0 --index-url https://download.pytorch.org/whl/test/${CUDA}`.

No matrix changes needed — the `X.Y.0` entries added in Stage 1 now resolve to
the RC.

## Stage 3: PyTorch X.Y final release

Examples: PR #647 (after an RC stage), PR #606 / #536 (single-shot, when
Stages 1–2 were skipped).

1. In all three install sites: delete the `X.Y.0` test-index `elif` (it falls
   through to the stable `else` branch). If Stage 2 was skipped, instead move
   the nightly `if` from `X.Y.0` to `X.(Y+1).0` now.
2. `.github/actions/setup/action.yml`: bump the default `torch-version` to
   `X.Y.0`.
3. Matrices: ensure `X.Y.0` is in all three `full_matrix_*.json` with the
   final CUDA set (drop CUDA variants PyTorch dropped, e.g. 2.12 dropped
   `cu128`); bump `minimal_matrix_*.json` to test `X.Y.0`.
4. `README.md`:
   - Update the `${TORCH}` version list and, if CUDA versions changed, the
     `${CUDA}` list.
   - Rename the "PyTorch Nightly" table to "PyTorch X.Y" and tick macOS `cpu`
     (PR #669 shows the exact edit).
5. `docs/requirements.txt`: bump the pinned torch CPU wheel URL.
6. `.github/workflows/linting.yml`: bump the torch pin in the ty type-check
   job.
7. `CHANGELOG.md`: `- Added PyTorch X.Y support ([#NNN](https://github.com/pyg-team/pyg-lib/pull/NNN))`.
8. Also update the example version strings in the `build_*.sh` usage error
   messages when convenient (PR #606 did).

Note: **final** pyg-lib wheels for `torch-X.Y.0` only appear at
https://data.pyg.org/whl once a pyg-lib release is cut with `X.Y.0` in the
full matrices (`release.yml` → workflow_dispatch → `release-type: final`), so
the README stable table must be accurate by the next "Prepare release" PR at
the latest.

## Stage 4: PyTorch patch release (X.Y.z)

Example: PR #687. Patch releases are ABI-compatible, so the same wheels are
served — just alias the index. In **both**
`.github/workflows/aws/upload_final_index.py` and
`.github/workflows/aws/upload_nightly_index.py`, add to the existing chain:

```python
if 'X.Y.0' in torch_version:
    wheels_dict[torch_version.replace('X.Y.0', 'X.Y.z')].append(wheel)
```

This makes `torch-X.Y.z+${CUDA}.html` resolve on data.pyg.org. No CHANGELOG
entry needed if trivial (use the `skip-changelog` label), but past PRs
included one.

## Stage 5: Drop an old PyTorch version

Examples: PR #681, #628, #557. Keep roughly the 3 newest stable versions.
Separate PR from adding support.

1. Remove the version's entries from all three `full_matrix_*.json`.
2. Rotate `minimal_matrix_*.json` so PR CI still covers newest + oldest
   supported.
3. `README.md`: remove its support table and drop it from the `${TORCH}` list.
4. `CHANGELOG.md`: entry under `### Removed`.

Do NOT touch the index-alias chains in `upload_*_index.py` — old wheels stay
served for users on old versions.

## New CUDA version (when PyTorch adds one)

Often a prerequisite for Stage 1. Examples: PR #614 (Docker), #630 (Windows),
#616 (arch lists).

- Linux builds use CUDA-pre-installed images
  `ghcr.io/pyg-team/pyg-lib/manylinux_2_28_${ARCH}:${CUDA_VERSION}`: add the
  new `cuda-version`/`cuda-number` to the matrix in
  `.github/workflows/_build_docker.yml` and teach
  `.github/workflows/docker/install_cuda.sh` (and `Dockerfile` if needed) the
  new version, then run the Docker workflow to publish images.
- Windows: add a `case` arm to `.github/workflows/cuda/Windows.sh` (installer
  URL) and `.github/workflows/cuda/Windows-env.sh` (PATH +
  `TORCH_CUDA_ARCH_LIST`).
- Keep `TORCH_CUDA_ARCH_LIST` in sync with the archs PyTorch's official wheels
  target for that CUDA version (see PR #616), and remember newer CUDA drops
  old archs (e.g. CUDA 13 dropped `sm_50`–`sm_70` support).

## Verifying a PR

1. Add the PR labels for the platforms touched: `os: linux`, `os: macos`,
   `os: windows`; add `ci: full` to run the full matrix when the change
   warrants it.
2. Alternatively trigger `pull.yml` via workflow_dispatch with
   `trigger-<os>: true` (and `full-matrix: true` for the full set):
   `gh workflow run pull.yml --ref <branch> -f trigger-linux=true`
3. After merge, confirm the next nightly `release.yml` run is green and the
   new `torch-X.Y.0+*.html` indices appear under
   https://data.pyg.org/whl/nightly/.
