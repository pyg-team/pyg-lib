---
name: support-new-pytorch
description: Add or update PyTorch version support in pyg-lib CI, wheels, and docs. Use when a new PyTorch version reaches any release stage — nightly, RC (test index), final release, or patch release (e.g. "support PyTorch 2.13", "2.13 RC is out", "add the 2.13.1 index") — or when dropping support for an old PyTorch version.
---

# Support a new PyTorch version

pyg-lib builds C++/CUDA wheels against specific PyTorch versions. Supporting a
new version is a staged process that follows PyTorch's own release lifecycle;
each stage is a small, self-contained change set (historically one PR each).
This skill contains every decision rule needed: complete the task WITHOUT
asking the user anything. Collect unavoidable judgment calls in an
"Assumptions" section of your final report instead of asking.

## Ground rules

- Do exactly one stage per invocation — the one matching the request.
- NEVER drop an old PyTorch version unless explicitly asked (that is Stage 5,
  its own task).
- Do not commit, push, or open a PR unless the request says to.
- Mirror the existing style of each file you touch (flags, quoting, comment
  and table formatting). When adding a matrix entry or README table, copy the
  entry/table of the previous newest version and edit only the values that
  change.
- Do not block on network access. Web checks below are optional
  cross-checks; if they fail or are unavailable, use the documented default
  and record the assumption.
- Missing-information defaults (apply silently, then record under
  Assumptions):
  - Python versions for a new torch version: same list as the previous newest
    stable entry in the same matrix file.
  - CUDA versions for a new torch version: same list as the previous newest
    stable entry, unless the request names CUDA versions that were added or
    dropped. (PyTorch's actual per-release CUDA set is at
    https://pytorch.org/get-started/locally/ — check only if network works.)
  - CHANGELOG PR number: if you are opening the PR yourself, create it first
    (or amend after creation) so the real number is used. Otherwise predict
    the next number as
    `gh api 'repos/pyg-team/pyg-lib/issues?state=all&per_page=1' --jq '.[0].number'`
    plus 1; if `gh`/network is unavailable, take the highest `(#N)` from
    `git log --oneline -15` plus 1. Either way, note in Assumptions that the
    number must be corrected when the PR is opened.

## Step 0: Determine the release stage

Trust the stage stated in the request. Keyword mapping:

| Request says                                                 | Stage |
| ------------------------------------------------------------ | ----- |
| "nightly", "start building against X.Y nightlies"            | 1     |
| "RC", "release candidate", "test index"                      | 2     |
| "X.Y (was) released", "final", or just "support PyTorch X.Y" | 3     |
| a patch version `X.Y.z` with z ≥ 1, "add the X.Y.z index"    | 4     |
| "drop", "remove", "deprecate" an old version                 | 5     |

Only if the request genuinely does not say: check
https://pypi.org/project/torch/#history (final versions appear there;
`X.Y.0` only on https://download.pytorch.org/whl/test = RC;
`X.Y.0.devYYYYMMDD` only on https://download.pytorch.org/whl/nightly =
nightly). Without network access, default to Stage 3 (final) and record the
assumption.

Independently, determine whether this PyTorch version introduces a **new CUDA
version** not yet present in
`.github/workflows/utils/full_matrix_linux.json`. The request usually says so
(e.g. "2.12 adds cu132"). If it does, the "New CUDA version" section below is
a prerequisite before that CUDA version may appear in any matrix. If nothing
indicates a new CUDA version, keep the previous stable's CUDA set.

## Key convention: the rolling pre-release placeholder

The **highest** torch version referenced in CI is a placeholder meaning
"pre-release" and is special-cased in exactly three install sites:

1. `.github/actions/setup/action.yml` — composite setup action (used by
   `_build_windows.yml`, `cpp_testing.yml`, `documentation.yml`). Also holds
   the default `torch-version` input, which must equal the current stable.
   Uses `uv pip install --no-config` style.
1. `.github/workflows/utils/build_linux.sh` — sets `CIBW_BEFORE_BUILD` /
   `CIBW_BEFORE_TEST` for cibuildwheel. Uses plain `pip install` inside the
   quoted CIBW variables.
1. `.github/workflows/utils/build_macos.sh` — same for macOS (CUDA index is
   always `cpu` there).

The special-cased branch installs from a different PyTorch index per stage:

- nightly: `--pre torch --index-url https://download.pytorch.org/whl/nightly/${CUDA}`
- RC: `torch==X.Y.0 --index-url https://download.pytorch.org/whl/test/${CUDA}`
- stable: `torch==X.Y.0 --index-url https://download.pytorch.org/whl/${CUDA}`
  (this is the plain `else` branch — a stable version needs NO special case)

When `X.Y` goes final, the placeholder version in the `if` conditions rolls
forward to `X.(Y+1).0`. Keep the placeholder version literally identical
across all three files. Keep any extra lines in the nightly branch (e.g. the
`uv pip install packaging` workaround in `action.yml`) unless the request
says to remove them.

## Build matrices

Six files: `.github/workflows/utils/{full,minimal}_matrix_{linux,macos,windows}.json`

- `full_*`: used by wheel releases (`release.yml`: nightly cron daily at 4:00
  UTC, and final releases via workflow_dispatch) and by PRs labeled
  `ci: full`. Lists every supported torch version.
- `minimal_*`: default PR CI. Exactly 2 entries: the newest version (with only
  its newest CUDA variant, e.g. `["cu132"]` on Linux/Windows, `["cpu"]` on
  macOS) and the oldest supported version (with one older CUDA variant, e.g.
  `["cu126"]`). Python list stays `["3.10", "3.14"]` (oldest + newest).
- Linux entries additionally carry `"arch": ["x86_64", "aarch64"]`; macOS and
  Windows entries do not have an `arch` key.
- Order entries newest-version-first in every file.
- PR builds only run when the PR has `os: linux` / `os: macos` /
  `os: windows` labels (`pull.yml` reads labels).

## Stage 1: Build against PyTorch X.Y nightlies

Usually started to get early signal and/or to ship wheels for a new CUDA
version. Prerequisite: the placeholder `if` branches already point at `X.Y.0`
(that happened when `X.(Y-1)` went final). If they still point at `X.Y.0`'s
predecessor, roll them forward first.

1. Add an `X.Y.0` entry at the top of `full_matrix_linux.json` and replace the
   newest entry of `minimal_matrix_linux.json` with it. Start narrow — newest
   CUDA + newest Python only (example: PR #611 added `2.12.0` with
   `["cu132"]` × `["3.14"]` × `["x86_64"]`) — and widen in follow-up requests
   once green (PR #629 expanded CUDA; #630 added Windows; #637 added macOS).
   If the request explicitly asks for Windows or macOS nightly support, edit
   those matrix files the same way (Windows may first need the "New CUDA
   version" steps).
1. `README.md`: add a "PyTorch Nightly" support table above the newest stable
   table, same column/row format, ticking only the built combinations.
1. `CHANGELOG.md`: add a line under `## [Unreleased]` → `### Added`, e.g.
   `- Added support for PyTorch \`X.Y.0+cuNNN\` wheels in nightly releases ([#N](https://github.com/pyg-team/pyg-lib/pull/N))`. (CI enforces a CHANGELOG entry on every PR unless it has the `skip-changelog\` label.)

Nightly pyg-lib wheels then appear under
https://data.pyg.org/whl/nightly/torch-X.Y.0+${CUDA}.html after the next
nightly `release.yml` run.

## Stage 2: PyTorch X.Y RC (test index)

Example: PR #641. In all three install sites, in this order:

1. Change the nightly `if` condition from `X.Y.0` to `X.(Y+1).0` (the
   placeholder rolls forward; PyTorch nightlies are now versioned
   `X.(Y+1).0.dev*`).
1. Add an `elif` for `X.Y.0` that installs from the test index, mirroring the
   file's style, with a comment like
   `# X.Y.0 is currently a release candidate; install from the test index.`:
   - `action.yml`: `uv pip install --no-config torch==${{ inputs.torch-version }} --index-url https://download.pytorch.org/whl/test/${{ inputs.cuda-version }}`
   - `build_linux.sh`: both CIBW vars get `pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/test/${CUDA_VERSION}`
   - `build_macos.sh`: same with the literal `cpu` index.
1. `CHANGELOG.md` entry under `### Changed` (or reuse/extend the existing
   `X.Y` line if one is already in `[Unreleased]`).

No matrix changes — the `X.Y.0` entries added in Stage 1 now resolve to the
RC. (If Stage 1 was skipped entirely, also add matrix entries as in Stage 1.)

## Stage 3: PyTorch X.Y final release

Examples: PR #647 (when an RC stage preceded), PR #606 / #536 (single-shot,
when Stages 1–2 were skipped). Apply ALL of the following; several may
already be done if earlier stages ran — verify each and skip only what is
already in place.

1. Install sites (all three): `X.Y.0` must NOT be special-cased anymore — it
   falls through to the stable `else` branch.
   - If an `X.Y.0` test-index `elif` exists (Stage 2 ran): delete it.
   - If the nightly `if` still points at `X.Y.0` (Stages 1–2 skipped): change
     it to `X.(Y+1).0`.
1. `.github/actions/setup/action.yml`: set the default `torch-version` input
   to `X.Y.0`.
1. Full matrices: ensure `X.Y.0` is the top entry in ALL THREE
   `full_matrix_*.json`, copying the previous newest entry's Python list;
   CUDA list per the defaults rule (macOS is always `["cpu"]`; Linux keeps its
   `arch` key). Drop CUDA variants only if the request says PyTorch dropped
   them.
1. Minimal matrices: in all three `minimal_matrix_*.json`, set the newest
   entry's `torch-version` to `X.Y.0` (keep its narrow CUDA list, updating it
   only if a new newest CUDA version was introduced). Leave the oldest entry
   alone.
1. `README.md`:
   - Add `X.Y.0` to the `${TORCH}` bullet list (keep existing versions; do
     not remove any).
   - Update the `${CUDA}` bullet list if the CUDA set changed (it is the
     union across the supported stable tables).
   - Support tables: if a "PyTorch Nightly" table exists for this version,
     rename its header to `PyTorch X.Y` and tick macOS `cpu` (see PR #669);
     otherwise insert a new `PyTorch X.Y` table above the previous newest
     one, copied from it, with columns matching the new CUDA list. Keep the
     older versions' tables.
1. `docs/requirements.txt`: bump the pinned torch CPU wheel URL to
   `https://download.pytorch.org/whl/cpu/torch-X.Y.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl`
   (substitute the version; keep the rest of the pattern; adjust `cp310` only
   if the repo's minimum Python changed).
1. Sweep for stray stable-version pins: `grep -rn "torch==" .github/workflows/ docs/`
   and bump any pinned stable torch version outside the three install sites
   (e.g. a type-check job in `linting.yml` has carried a pin on some
   branches). If the grep only matches the install sites and
   `docs/requirements.txt`, there is nothing extra to do.
1. `CHANGELOG.md`: `- Added PyTorch X.Y support ([#N](https://github.com/pyg-team/pyg-lib/pull/N))`
   under `## [Unreleased]` → `### Added`.
1. Optional (matches past PRs, do it): refresh the example versions in the
   `:?Specify torch version, e.g. ...` usage strings of `build_linux.sh` /
   `build_macos.sh` to `X.Y.0`.

Note: final pyg-lib wheels for `torch-X.Y.0` only appear at
https://data.pyg.org/whl once a pyg-lib release is cut with `X.Y.0` in the
full matrices (`release.yml` → workflow_dispatch → `release-type: final`).
Nightly wheels appear after the next nightly cron run.

## Stage 4: PyTorch patch release (X.Y.z, z ≥ 1)

Example: PR #687. Patch releases are ABI-compatible, so the existing wheels
are simply aliased in the wheel indices. In **both**
`.github/workflows/aws/upload_final_index.py` and
`.github/workflows/aws/upload_nightly_index.py`, extend the existing `if`
chain, mirroring the neighbors exactly:

```python
if 'X.Y.0' in torch_version:
    wheels_dict[torch_version.replace('X.Y.0', 'X.Y.z')].append(wheel)
```

(If `X.Y.1` already has an alias and `X.Y.2` came out, add a second
`.replace(...)` line inside the same `if`, like the `2.1.0`/`2.2.0` blocks.)
This makes `torch-X.Y.z+${CUDA}.html` resolve on data.pyg.org. No other
files change, and no CHANGELOG entry is added (matching PR #687) — instead
state in your report that the PR needs the `skip-changelog` label and no
`os:` labels.

## Stage 5: Drop an old PyTorch version

Examples: PR #681, #628, #557. Keep roughly the 3 newest stable versions.
Only do this when explicitly asked.

1. Remove the version's entries from all three `full_matrix_*.json`.
1. If the dropped version was the oldest entry in any `minimal_matrix_*.json`,
   replace that entry's `torch-version` with the now-oldest supported stable
   (keep the narrow CUDA list, adjusting to a CUDA version that oldest
   supports).
1. `README.md`: delete its support table and remove it from the `${TORCH}`
   bullet list; shrink the `${CUDA}` list if a CUDA version is no longer used
   by any remaining table.
1. `CHANGELOG.md`: `- Dropped support for PyTorch X.Y` under `### Removed`.

Do NOT touch the alias chains in `upload_*_index.py` — already-published
wheels stay served for users on old versions.

## New CUDA version (when PyTorch adds one)

Prerequisite for putting that CUDA version in any matrix. Examples: PR #614
(Docker), #630 (Windows), #616 (arch lists). Skip any part already present.

- Linux builds use CUDA-pre-installed images
  `ghcr.io/pyg-team/pyg-lib/manylinux_2_28_${ARCH}:${CUDA_VERSION}`: add the
  new `cuda-version` (`cuNNN`) + `cuda-number` (`"NN.N"`) pair to the matrix
  in `.github/workflows/_build_docker.yml`, and teach
  `.github/workflows/docker/install_cuda.sh` (and `Dockerfile` if needed) the
  new version. Note in your report that the Docker workflow must be run once
  (workflow_dispatch) to publish images before Linux CI can use the new CUDA.
- Windows: add a `case` arm to `.github/workflows/cuda/Windows.sh` (installer
  URL under https://developer.download.nvidia.com/compute/cuda/) and to
  `.github/workflows/cuda/Windows-env.sh` (PATH + `TORCH_CUDA_ARCH_LIST`),
  copying the previous newest arm.
- Keep `TORCH_CUDA_ARCH_LIST` matched to the archs PyTorch's official wheels
  target for that CUDA version (PR #616); newer CUDA drops old archs (CUDA 13
  dropped Maxwell/Pascal/Volta, so lists start at `7.5`).

## Self-check before finishing (mandatory)

Run and confirm each item that applies to the stage you executed:

```bash
for f in .github/workflows/utils/*_matrix_*.json; do jq -e . "$f" >/dev/null || echo "BROKEN: $f"; done
grep -n "== ['\"]*2\." .github/actions/setup/action.yml .github/workflows/utils/build_linux.sh .github/workflows/utils/build_macos.sh
```

- The version compared in the special-case branches is identical across all
  three install sites, and equals the highest version anywhere in CI.
- Stage 3 only: the new stable `X.Y.0` appears in all six matrix files, is
  the `torch-version` default in `action.yml`, is in the README `${TORCH}`
  list with its own table, is in `docs/requirements.txt`, and is pinned in
  `linting.yml`; and `X.Y.0` appears in NO `if`/`elif` condition of the three
  install sites.
- Stage 4 only: both `upload_*_index.py` files gained the same alias.
- `CHANGELOG.md` has a new entry under `## [Unreleased]` (required by the
  changelog-enforcer CI unless the PR gets the `skip-changelog` label).
- `git diff --stat` matches the stage's expected file set — nothing
  unrelated changed.

## Final report format

End with: (1) the stage executed and why; (2) `git diff --stat`; (3) one line
per changed file; (4) an "Assumptions" list (empty if none); (5) remaining
manual steps (e.g. "run the Docker workflow", "PR labels: os: linux,
os: windows, ci: full", "correct the CHANGELOG PR number after opening the
PR").

## Only when asked to open/verify a PR

1. Branch, commit, push, `gh pr create`; then fix the CHANGELOG PR number to
   the real one and push again.
1. Add labels for the platforms touched (`os: linux`, `os: macos`,
   `os: windows`) plus `ci: full` when the change warrants the full matrix:
   `gh pr edit <num> --add-label "os: linux" --add-label "ci: full"`
1. Builds can also be triggered without labels:
   `gh workflow run pull.yml --ref <branch> -f trigger-linux=true -f full-matrix=true`
1. After merge, the next nightly `release.yml` run should be green and new
   `torch-X.Y.0+*.html` indices appear under https://data.pyg.org/whl/nightly/.
