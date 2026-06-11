# Branch: `server-block-diffusion`

An out-of-tree `tools/server` integration for block-diffusion (canvas) models
— concretely, DiffusionGemma as implemented by ggml-org/llama.cpp
[PR #24423](https://github.com/ggml-org/llama.cpp/pull/24423).

`llama-server` on this branch serves DiffusionGemma GGUFs through the
existing HTTP endpoints (`/completion`, `/v1/chat/completions`) with **zero
API-surface change**. Autoregressive models take byte-identical code paths;
the single divergence point is in `update_slots`, gated on the GGUF key
`diffusion.canvas_length`. The denoiser is the unmodified
`examples/diffusion/diffusion.cpp` from PR #24423 (danielhanchen's code,
linked in — not reimplemented).

**Status: this is a branch, not a pull request.** Nothing here is submitted
to ggml-org while no diffusion model core has merged. It exists to timestamp
authorship of the server mode and to let interested people build and test it.
If PR #24423 (or another core) merges, this work will be rebased and offered
upstream in whatever shape the llama.cpp maintainers prefer.

## Provenance

This branch = the head of PR #24423 + exactly three server commits.

| Commit | Author | Role |
|---|---|---|
| `d2462f8f7ac6d8` and below | ggml-org contributors | upstream `master` history (merge-base with PR #24423) |
| `c5fe75b9765965` | danielhanchen | PR #24423: diffusion-visual updates |
| `c84e85af61011f` | danielhanchen | PR #24423: Windows build fix, test-llama-archs skip, drop debug hooks |
| `7c200ddb0b97e6` | danielhanchen | PR #24423: diffusion-cli `--fit` note |
| `9b4beb7edf56bf` | danielhanchen | PR #24423: diffusion-cli `-ot` / `--n-cpu-moe` |
| `15ad8f4201d05f` | danielhanchen | PR #24423: device-resident self-conditioning + CLI throughput |
| `d6cf0b288dfac7` | danielhanchen | PR #24423: device-side sampling reductions (default on) |
| `53752ade13c86e` | danielhanchen | PR #24423 head: HIP/MUSA build fix for the diffusion sampler |
| `4182c87c1e3378` | Audrey Tang | server: block-diffusion generation mode for diffusion-gemma |
| `f773a0c2382951` | Audrey Tang | server: cap diffusion ubatch growth, require explicit `-ub` for long prompts |
| `3b45f0e4a6c464` | Audrey Tang | server: extract final answer after channel scaffold for raw completion path |
| (tip) | Audrey Tang | this BRANCH-README.md — documentation only |

The branch carries PR #24423's unmerged history with git authorship intact —
danielhanchen's commits remain his; audreyt's additions are exactly the
listed SHAs. Net diff over the PR #24423 head: 5 files, +530/−0
(`tools/server/server-context.cpp`, `tools/server/server-task.{h,cpp}`,
`tools/server/CMakeLists.txt`, `common/common.cpp`).

Not on this branch: any code from PR #24427 (lnigam's parallel
implementation), and no Ollama-specific code (see the `dg-preview` branch
for that).

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-diffusion-cli -j
```

Metal is enabled by default on Apple Silicon; nothing diffusion-specific is
needed at configure time. Run against a DiffusionGemma GGUF that follows
PR #24423's conventions (arch `diffusion-gemma`, GGUF key
`diffusion.canvas_length`):

```sh
build/bin/llama-server -m diffusiongemma.gguf --port 8080
```

The server forces `n_parallel=1` for diffusion models (request state lives on
the model object) and defaults the context to 8192 to keep the
`[prompt | canvas]` ubatch tractable; prompts longer than
`n_ubatch − canvas_length` need an explicit `-ub`.

## Verification

Verified 2026-06-11 on Apple M5 Max (128 GB, Metal), DiffusionGemma Q4_K_M,
Release build, at this branch's exact commits:

- **Oracle byte-equivalence:** with the identical templated prompt, seed 42,
  `n_predict 256`, `llama-diffusion-cli` and `llama-server` `/completion`
  (`diffusion_raw: true`) produce byte-identical output — under BOTH the
  PR's new device-sampling default (`--diffusion-gpu-sample-reduce` auto/on)
  and explicitly off. The server pins the host byte-exact sampling path
  (`gpu_sampling=false`), so the new default does not alter server output.
- **Throughput (256-token canvas, 19 denoise steps):** CLI 76.9/76.2 tok/s
  (device sampling on), 75.7/75.7 (off); server 75.7/76.3 non-streamed,
  66.6 streamed.
- **Scaffold extraction** (`3b45f0e4`): raw `/completion` responses strip the
  `<|channel>thought … <channel|>` scaffold; when no close-tag appears, the
  raw canvas is flushed unmodified.
- **AR regression:** non-diffusion models are unaffected — the diffusion path
  is gated on `diffusion.canvas_length` and never entered otherwise.

## Licence and disclosure

MIT, same as upstream llama.cpp. This README and the verification harness
were prepared with AI assistance; the server-mode code and all upstream
communication are authored and reviewed by Audrey Tang.
