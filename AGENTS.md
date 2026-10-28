# ufo-cuda — agent notes

Chez Scheme FFI bindings for the CUDA driver API. Pure-Scheme R6RS
libraries, zero dependencies, published to Akku as `ufo-cuda`.

## Layout

- `ufo/cuda/ffi.sls` — `(ufo cuda ffi)`: raw bindings, created lazily by
  `load-cuda-driver!`. Symbol names verified against
  `stubs/libcuda.so` (CUDA 12.9): versioned `_v2` entry points are bound
  directly because `cuda.h`'s unversioned names are macros.
- `ufo/cuda.sls` — `(ufo cuda)`: high-level API, `&cuda-error` condition
  type, CUresult table.
- `ufo/cuda/kernel.sls` — `(ufo cuda kernel)`: `cuda-launch!`,
  `cuda-compile-cubin`, `cuda-device-gencode`.
- `tests/test-basic.sps` — Tier-1 tests (driver stub, no GPU needed).
- `examples/vecadd.{cu,ss}` — Tier-3 vector addition example (needs GPU).

## Build / test

- No build step. Run tests with:
  `scheme --libdirs . --script tests/test-basic.sps`
- Dev environment: `nix develop` (provides chez + cudatoolkit and sets
  `$UFO_CUDA_LIB` to the driver stub). The flake pins nixpkgs to a
  machine-local store path; `flake.lock` is gitignored, regenerate per
  machine.

## Conventions and pitfalls

- This machine has **no NVIDIA GPU** — never claim GPU execution works;
  Tier-3 (`examples/vecadd.ss`) must be run elsewhere.
- Chez R6RS quirks encountered here:
  - exporting a variable that is `set!` in the library body is rejected
    (ffi layer therefore stores foreign procedures in `let`-bound cells);
  - a record type name is itself a syntactic binding, so no procedure may
    share it (query proc is `cuda-get-device`, record type `cuda-device`);
  - `let` initializers cannot refer to the let's own variables (use
    `let*`).
- Keep the library zero-dependency (Akku archive requires packages to be
  self-sufficient).
- When adding driver API bindings, verify the exported symbol with
  `nm -D $libcuda` first; prefer `_v2` suffixed names.
