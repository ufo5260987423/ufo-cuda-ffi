#!r6rs ; -*- mode: scheme; coding: utf-8 -*-
;;; ufo/cuda/ffi.sls — Raw Chez Scheme FFI bindings for the CUDA driver API.
;;;
;;; This library is deliberately thin: one procedure per C function, with
;;; C types mapped onto Chez foreign types and no error checking.  All
;;; error handling and resource management lives in (ufo cuda).
;;;
;;; The exported cu-* names are procedures that forward to the actual
;;; foreign procedures.  The foreign procedures are created by
;;; load-cuda-driver!, so the library can be loaded before (or after) the
;;; shared object exists on disk.  Calling a cu-* procedure before
;;; load-cuda-driver! raises an &assertion.
;;;
;;; Note on symbol names: cuda.h exports several versioned entry points
;;; (e.g. cuMemAlloc is a macro that expands to cuMemAlloc_v2).  The _v2
;;; symbols are what libcuda actually exports, so they are bound here
;;; directly.
(library (ufo cuda ffi)
  (export
    load-cuda-driver!
    cu-init cu-driver-get-version
    cu-device-get cu-device-get-count cu-device-get-name
    cu-device-get-attribute
    cu-device-primary-ctx-retain cu-device-primary-ctx-release
    cu-ctx-set-current cu-ctx-get-current cu-ctx-synchronize
    cu-mem-alloc cu-mem-free
    cu-memcpy-host-to-device! cu-memcpy-device-to-host!
    cu-memcpy-device-to-device! cu-memset-d8!
    cu-module-load-data cu-module-unload cu-module-get-function
    cu-launch-kernel
    cu-stream-create cu-stream-destroy cu-stream-synchronize
    cu-get-error-name cu-get-error-string)

  (import (chezscheme))

  ;; Thunks that create the foreign procedures; populated when each
  ;; define-entry below is evaluated, and run by load-cuda-driver!.
  (define bindings '())

  (define (register-binding! name thunk)
    (set! bindings (cons (cons name thunk) bindings)))

  (define-syntax define-entry
    (syntax-rules ()
      [(_ public c-name (arg ...) ret)
       (define public
         ;; Each expansion gets its own hygienic cell for `raw'.
         (let ((raw #f))
           (register-binding!
            'public
            (lambda ()
              (set! raw (foreign-procedure c-name (arg ...) ret))))
           (lambda args
             (if raw
                 (apply raw args)
                 (assertion-violation 'public
                                      "CUDA driver not loaded; call load-cuda-driver! first"
                                      #f)))))]))

;;; ---------------------------------------------------------------
;;; C signatures (from cuda.h, CUDA 12.x):
;;;
;;; CUresult cuInit(unsigned int Flags)
;;; CUresult cuDriverGetVersion(int *driverVersion)
;;; CUresult cuDeviceGet(CUdevice *device, int ordinal)      ; CUdevice = int
;;; CUresult cuDeviceGetCount(int *count)
;;; CUresult cuDeviceGetName(char *name, int len, CUdevice dev)
;;; CUresult cuDeviceGetAttribute(int *value, int attrib, CUdevice dev)
;;; CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev)
;;; CUresult cuDevicePrimaryCtxRelease(CUdevice dev)
;;; CUresult cuCtxSetCurrent(CUcontext ctx)
;;; CUresult cuCtxGetCurrent(CUcontext *pctx)
;;; CUresult cuCtxSynchronize(void)
;;; CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
;;; CUresult cuMemFree_v2(CUdeviceptr dptr)
;;; CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t n)
;;; CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t n)
;;; CUresult cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t n)
;;; CUresult cuMemsetD8_v2(CUdeviceptr dst, unsigned char value, size_t n)
;;; CUresult cuModuleLoadData(CUmodule *module, const void *image)
;;; CUresult cuModuleUnload(CUmodule hmod)
;;; CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
;;;                              const char *name)
;;; CUresult cuLaunchKernel(CUfunction f,
;;;                         unsigned gx, unsigned gy, unsigned gz,
;;;                         unsigned bx, unsigned by, unsigned bz,
;;;                         unsigned sharedMemBytes, CUstream hStream,
;;;                         void **kernelParams, void **extra)
;;; CUresult cuStreamCreate(CUstream *phStream, unsigned Flags)
;;; CUresult cuStreamDestroy(CUstream hStream)
;;; CUresult cuStreamSynchronize(CUstream hStream)
;;; CUresult cuGetErrorName(CUresult error, const char **pStr)
;;; CUresult cuGetErrorString(CUresult error, const char **pStr)
;;; ---------------------------------------------------------------

  (define-entry cu-init "cuInit" (unsigned) int)
  (define-entry cu-driver-get-version "cuDriverGetVersion" (void*) int)

  (define-entry cu-device-get "cuDeviceGet" (void* int) int)
  (define-entry cu-device-get-count "cuDeviceGetCount" (void*) int)
  (define-entry cu-device-get-name "cuDeviceGetName" (void* int int) int)
  (define-entry cu-device-get-attribute "cuDeviceGetAttribute" (void* int int) int)

  (define-entry cu-device-primary-ctx-retain "cuDevicePrimaryCtxRetain" (void* int) int)
  (define-entry cu-device-primary-ctx-release "cuDevicePrimaryCtxRelease" (int) int)
  (define-entry cu-ctx-set-current "cuCtxSetCurrent" (void*) int)
  (define-entry cu-ctx-get-current "cuCtxGetCurrent" (void*) int)
  (define-entry cu-ctx-synchronize "cuCtxSynchronize" () int)

  (define-entry cu-mem-alloc "cuMemAlloc_v2" (void* unsigned-64) int)
  (define-entry cu-mem-free "cuMemFree_v2" (unsigned-64) int)
  (define-entry cu-memcpy-host-to-device! "cuMemcpyHtoD_v2" (unsigned-64 void* unsigned-64) int)
  (define-entry cu-memcpy-device-to-host! "cuMemcpyDtoH_v2" (void* unsigned-64 unsigned-64) int)
  (define-entry cu-memcpy-device-to-device! "cuMemcpyDtoD_v2" (unsigned-64 unsigned-64 unsigned-64) int)
  (define-entry cu-memset-d8! "cuMemsetD8_v2" (unsigned-64 unsigned-8 unsigned-64) int)

  (define-entry cu-module-load-data "cuModuleLoadData" (void* void*) int)
  (define-entry cu-module-unload "cuModuleUnload" (void*) int)
  (define-entry cu-module-get-function "cuModuleGetFunction" (void* void* string) int)

  (define-entry cu-launch-kernel "cuLaunchKernel"
    (void* unsigned unsigned unsigned unsigned unsigned unsigned unsigned void* void* void*) int)

  (define-entry cu-stream-create "cuStreamCreate" (void* unsigned) int)
  (define-entry cu-stream-destroy "cuStreamDestroy" (void*) int)
  (define-entry cu-stream-synchronize "cuStreamSynchronize" (void*) int)

  (define-entry cu-get-error-name "cuGetErrorName" (int void*) int)
  (define-entry cu-get-error-string "cuGetErrorString" (int void*) int)

  (define (load-cuda-driver! path)
    (unless (string? path)
      (assertion-violation 'load-cuda-driver! "expected a string path" path))
    (let ((handle (load-shared-object path)))
      ;; Bind in declaration order.
      (for-each (lambda (binding) ((cdr binding))) (reverse bindings))
      handle)))
