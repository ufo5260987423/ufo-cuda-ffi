# ufo-cuda

CUDA driver API bindings for Chez Scheme, via the FFI (`load-shared-object` +
`foreign-procedure`). Kernels are compiled ahead of time with `nvcc` and
loaded at runtime — the library itself is pure Scheme with zero dependencies.

## Libraries

| Library           | Contents                                                   |
|-------------------|------------------------------------------------------------|
| `(ufo cuda ffi)`  | Raw one-to-one bindings of ~27 driver API entry points.    |
| `(ufo cuda)`      | High-level API: devices, contexts, memory, modules, streams, `&cuda-error` conditions. |
| `(ufo cuda kernel)` | `cuda-launch!` (marshals Scheme values into `void** kernelParams`) and `nvcc` compilation helpers. |

## Requirements

- Chez Scheme (developed on 10.4)
- An NVIDIA CUDA driver (`libcuda.so`) at runtime for anything real; the
  CUDA toolkit's `stubs/libcuda.so` is enough for the test suite
- `nvcc` only for compiling kernels

The driver library path is resolved by, in order:

1. the argument to `(cuda-load! path)`,
2. the environment variable `$UFO_CUDA_LIB`,
3. default paths including `/run/opengl-driver/lib/libcuda.so.1` (NixOS)
   and common FHS locations.

## Quick start

```scheme
(import (ufo cuda) (ufo cuda kernel))

(cuda-load!)
(cuda-init!)
(define dev (cuda-get-device 0))
(define ctx (cuda-primary-context dev))

;; compile once per compute capability:
(cuda-compile-cubin "vecadd.cu" "vecadd.cubin" (cuda-device-gencode dev))

(cuda-with-context ctx
  (lambda ()
    (define mod (cuda-module-load-data (read-binary-file "vecadd.cubin")))
    (define vecadd (cuda-module-get-function mod "vecadd"))
    (define a (cuda-malloc (* 1024 4)))
    ...))
```

See `examples/vecadd.ss` for a complete vector-addition program.

Kernel parameters in `cuda-launch!` are `(ftype . value)` pairs plus bare
`<cuda-device-ptr>` records, e.g.:

```scheme
(cuda-launch! vecadd (list blocks 1 1) (list 256 1 1)
              (list a-dev b-dev c-dev (cons 'int n)))
```

## Development (NixOS, no GPU)

```sh
nix develop                 # provides chez + cudatoolkit, sets $UFO_CUDA_LIB
scheme --libdirs . --script tests/test-basic.sps
```

The dev shell sets `$UFO_CUDA_LIB` to the toolkit's driver stub. The stub
answers every call with `CUDA_ERROR_STUB_LIBRARY` (34), which the Tier-1
tests use to verify symbol resolution, error propagation and condition
fields without any GPU.

Test tiers:

- **Tier 1** (runs anywhere, incl. this GPU-less machine): FFI binding
  correctness, error handling, nvcc compilation pipeline.
  `tests/test-basic.sps`.
- **Tier 2**: pure-Scheme logic unit tests.
- **Tier 3** (needs an NVIDIA GPU + driver): end-to-end kernel execution,
  `examples/vecadd.ss` — it prints `PASS: c[i] == 3*i for all i` when the
  result is correct.

## Publishing to Akku

1. Finalize version/synopsis in `Akku.manifest`.
2. `git tag -s v0.1.0` (GPG-signed tag), push.
3. `akku publish`.
