#!r6rs ; -*- mode: scheme; coding: utf-8 -*-
;;; ufo/cuda.sls — High-level CUDA driver API for Chez Scheme.
;;;
;;; Error handling: every driver call that returns a non-success CUresult
;;; raises an &cuda-error condition carrying the numeric code, the name
;;; of the API entry point, and a human-readable message.
;;;
;;; The driver library is loaded lazily with cuda-load!.  When no path is
;;; given, $UFO_CUDA_LIB is consulted, followed by the usual locations of
;;; the NVIDIA driver (incl. the NixOS one).
(library (ufo cuda)
  (export
    ;; driver loading & initialization
    cuda-load! cuda-loaded? cuda-init! cuda-initialized?
    cuda-driver-version
    ;; errors
    &cuda-error make-cuda-error cuda-error? cuda-error-code cuda-error-api
    cuda-error-code->symbol cuda-error-symbol->code cuda-check
    ;; devices
    cuda-device? cuda-get-device cuda-device-count cuda-device-handle
    cuda-device-name cuda-device-attribute
    cuda-attribute-compute-capability-major
    cuda-attribute-compute-capability-minor
    cuda-attribute-multiprocessor-count
    cuda-device-compute-capability
    ;; contexts
    cuda-context? cuda-context-handle cuda-context-device
    cuda-primary-context cuda-context-destroy!
    cuda-current-context cuda-with-context cuda-device-sync!
    ;; device memory
    cuda-device-ptr? cuda-device-ptr cuda-device-ptr-address
    cuda-device-ptr-size cuda-malloc cuda-free! cuda-memset!
    cuda-memcpy-host->device! cuda-memcpy-device->host
    cuda-memcpy-device->device!
    ;; modules & functions
    cuda-module? cuda-module-handle cuda-module-load-data cuda-module-unload!
    cuda-function? cuda-function-handle cuda-function-module
    cuda-module-get-function
    ;; streams
    cuda-stream? cuda-stream-handle cuda-stream-default
    cuda-stream-create cuda-stream-destroy! cuda-stream-synchronize)

  (import (chezscheme)
          (ufo cuda ffi))

;;; ------------------------------------------------------------------
;;; Conditions
;;; ------------------------------------------------------------------

  (define-condition-type &cuda-error &error
    make-cuda-error cuda-error?
    (code cuda-error-code)
    (api cuda-error-api))

  ;; Code -> symbol table for the CUresult enum (CUDA 12.x).
  (define error-table
    '((0   . CUDA_SUCCESS)
      (1   . CUDA_ERROR_INVALID_VALUE)
      (2   . CUDA_ERROR_OUT_OF_MEMORY)
      (3   . CUDA_ERROR_NOT_INITIALIZED)
      (4   . CUDA_ERROR_DEINITIALIZED)
      (5   . CUDA_ERROR_PROFILER_DISABLED)
      (6   . CUDA_ERROR_PROFILER_NOT_INITIALIZED)
      (7   . CUDA_ERROR_PROFILER_ALREADY_STARTED)
      (8   . CUDA_ERROR_PROFILER_ALREADY_STOPPED)
      (34  . CUDA_ERROR_STUB_LIBRARY)
      (46  . CUDA_ERROR_DEVICE_UNAVAILABLE)
      (100 . CUDA_ERROR_NO_DEVICE)
      (101 . CUDA_ERROR_INVALID_DEVICE)
      (200 . CUDA_ERROR_INVALID_IMAGE)
      (201 . CUDA_ERROR_INVALID_CONTEXT)
      (202 . CUDA_ERROR_CONTEXT_ALREADY_CURRENT)
      (205 . CUDA_ERROR_MAP_FAILED)
      (206 . CUDA_ERROR_UNMAP_FAILED)
      (209 . CUDA_ERROR_NO_BINARY_FOR_GPU)
      (211 . CUDA_ERROR_NOT_MAPPED)
      (214 . CUDA_ERROR_ECC_UNCORRECTABLE)
      (218 . CUDA_ERROR_INVALID_PTX)
      (219 . CUDA_ERROR_INVALID_GRAPHICS_CONTEXT)
      (221 . CUDA_ERROR_JIT_COMPILER_NOT_FOUND)
      (222 . CUDA_ERROR_UNSUPPORTED_PTX_VERSION)
      (400 . CUDA_ERROR_INVALID_HANDLE)
      (500 . CUDA_ERROR_NOT_FOUND)
      (600 . CUDA_ERROR_NOT_READY)
      (700 . CUDA_ERROR_ILLEGAL_ADDRESS)
      (701 . CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES)
      (702 . CUDA_ERROR_LAUNCH_TIMEOUT)
      (703 . CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING)
      (704 . CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
      (705 . CUDA_ERROR_PEER_ACCESS_NOT_ENABLED)
      (708 . CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE)
      (709 . CUDA_ERROR_CONTEXT_IS_DESTROYED)
      (710 . CUDA_ERROR_ASSERT)
      (715 . CUDA_ERROR_ILLEGAL_INSTRUCTION)
      (716 . CUDA_ERROR_MISALIGNED_ADDRESS)
      (717 . CUDA_ERROR_INVALID_ADDRESS_SPACE)
      (718 . CUDA_ERROR_INVALID_PC)
      (719 . CUDA_ERROR_LAUNCH_FAILED)
      (800 . CUDA_ERROR_NOT_PERMITTED)
      (801 . CUDA_ERROR_NOT_SUPPORTED)
      (802 . CUDA_ERROR_SYSTEM_NOT_READY)
      (803 . CUDA_ERROR_SYSTEM_DRIVER_MISMATCH)
      (804 . CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE)
      (900 . CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED)
      (901 . CUDA_ERROR_STREAM_CAPTURE_INVALIDATED)
      (902 . CUDA_ERROR_STREAM_CAPTURE_MERGE)
      (903 . CUDA_ERROR_STREAM_CAPTURE_UNMATCHED)
      (909 . CUDA_ERROR_TIMEOUT)
      (999 . CUDA_ERROR_UNKNOWN)))

  (define (cuda-error-code->symbol code)
    (cond ((assv code error-table) => cdr)
          (else 'CUDA_ERROR_UNKNOWN)))

  (define (cuda-error-symbol->code sym)
    (let loop ((table error-table))
      (cond ((null? table) #f)
            ((eq? (cdar table) sym) (caar table))
            (else (loop (cdr table))))))

  (define (raise-cuda-error api code)
    (raise
     (condition
      (make-cuda-error code api)
      (make-message-condition
       (string-append (symbol->string (cuda-error-code->symbol code))
                      " (" (number->string code) ")"
                      " in " (symbol->string api))))))

  (define-syntax cuda-check
    (syntax-rules ()
      [(_ api expr)
       (let ((result expr))
         (unless (zero? result)
           (raise-cuda-error 'api result))
         result)]))

  ;; Internal alias, kept short for use inside this library.
  (define-syntax check-cuda (syntax-rules () [(_ . rest) (cuda-check . rest)]))

;;; ------------------------------------------------------------------
;;; Helpers for C memory
;;; ------------------------------------------------------------------

  ;; Allocate a foreign buffer of TYPE, run THUNK with its address, free it.
  (define (call-with-foreign-buffer type thunk)
    (let ((buf (foreign-alloc (foreign-sizeof type))))
      (dynamic-wind
        (lambda () #f)
        (lambda () (thunk buf))
        (lambda () (foreign-free buf)))))

  ;; Read a NUL-terminated C string from foreign memory.
  (define (foreign-c-string ptr)
    (let loop ((i 0) (chars '()))
      (let ((c (foreign-ref 'unsigned-8 ptr i)))
        (if (zero? c)
            (list->string (reverse chars))
            (loop (+ i 1) (cons (integer->char c) chars))))))

  ;; Copy a bytevector into freshly allocated foreign memory; returns
  ;; the new pointer.  Caller must free.
  (define (bytevector->foreign-memory bv)
    (let* ((n (bytevector-length bv))
           (buf (foreign-alloc (max n 1))))
      (let loop ((i 0))
        (if (< i n)
            (begin (foreign-set! 'unsigned-8 buf i (bytevector-u8-ref bv i))
                   (loop (+ i 1)))
            buf))))

;;; ------------------------------------------------------------------
;;; Driver loading & initialization
;;; ------------------------------------------------------------------

  (define default-driver-paths
    ;; NixOS NVIDIA driver location comes first; then common FHS paths.
    '("/run/opengl-driver/lib/libcuda.so.1"
      "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
      "/usr/lib/libcuda.so.1"
      "libcuda.so.1"))

  (define (find-driver-path)
    (cond ((getenv "UFO_CUDA_LIB") => values)
          (else
           (let loop ((paths default-driver-paths))
             (cond ((null? paths)
                    (assertion-violation 'cuda-load!
                      "libcuda.so not found; set $UFO_CUDA_LIB" #f))
                   ((file-exists? (car paths)) (car paths))
                   (else (loop (cdr paths))))))))

  (define driver-loaded? #f)
  (define driver-initialized? #f)

  (define cuda-load!
    (case-lambda
      (()
       (cuda-load! (find-driver-path)))
      ((path)
       (load-cuda-driver! path)
       (set! driver-loaded? #t)
       (void))))

  (define (cuda-loaded?) driver-loaded?)

  (define (ensure-loaded who)
    (unless driver-loaded?
      (assertion-violation who "CUDA driver not loaded; call (cuda-load!) first" #f)))

  (define (cuda-init!)
    (ensure-loaded 'cuda-init!)
    (check-cuda cu-init (cu-init 0))
    (set! driver-initialized? #t)
    (void))

  (define (cuda-initialized?) driver-initialized?)

  (define (ensure-initialized who)
    (unless driver-initialized?
      (assertion-violation who "CUDA driver not initialized; call (cuda-init!) first" #f)))

  (define (cuda-driver-version)
    (ensure-initialized 'cuda-driver-version)
    (call-with-foreign-buffer
     'int
     (lambda (buf)
       (check-cuda cu-driver-get-version (cu-driver-get-version buf))
       (foreign-ref 'int buf 0))))

;;; ------------------------------------------------------------------
;;; Devices
;;; ------------------------------------------------------------------

  (define-record-type (cuda-device make-cuda-device cuda-device?)
    (fields (immutable handle)))

  (define (cuda-device-count)
    (ensure-initialized 'cuda-device-count)
    (call-with-foreign-buffer
     'int
     (lambda (buf)
       (check-cuda cu-device-get-count (cu-device-get-count buf))
       (foreign-ref 'int buf 0))))

  (define (cuda-get-device ordinal)
    (ensure-initialized 'cuda-device)
    (call-with-foreign-buffer
     'int
     (lambda (buf)
       (check-cuda cu-device-get (cu-device-get buf ordinal))
       (make-cuda-device (foreign-ref 'int buf 0)))))

  (define (cuda-device-name dev)
    (let ((buf (foreign-alloc 256)))
      (dynamic-wind
        (lambda () #f)
        (lambda ()
          (check-cuda cu-device-get-name
                      (cu-device-get-name buf 256 (cuda-device-handle dev)))
          (foreign-c-string buf))
        (lambda () (foreign-free buf)))))

  ;; Selected CUdevice_attribute enum values.
  (define cuda-attribute-multiprocessor-count 16)
  (define cuda-attribute-compute-capability-major 75)
  (define cuda-attribute-compute-capability-minor 76)

  (define (cuda-device-attribute dev attrib)
    (call-with-foreign-buffer
     'int
     (lambda (buf)
       (check-cuda cu-device-get-attribute
                   (cu-device-get-attribute buf attrib (cuda-device-handle dev)))
       (foreign-ref 'int buf 0))))

  (define (cuda-device-compute-capability dev)
    (cons (cuda-device-attribute dev cuda-attribute-compute-capability-major)
          (cuda-device-attribute dev cuda-attribute-compute-capability-minor)))

;;; ------------------------------------------------------------------
;;; Contexts
;;; ------------------------------------------------------------------

  (define-record-type (cuda-context make-cuda-context cuda-context?)
    (fields (immutable handle) (immutable device)))

  (define (cuda-primary-context dev)
    (call-with-foreign-buffer
     'void*
     (lambda (buf)
       (check-cuda cu-device-primary-ctx-retain
                   (cu-device-primary-ctx-retain buf (cuda-device-handle dev)))
       (make-cuda-context (foreign-ref 'void* buf 0) dev))))

  (define (cuda-context-destroy! ctx)
    (check-cuda cu-device-primary-ctx-release
                (cu-device-primary-ctx-release
                 (cuda-device-handle (cuda-context-device ctx))))
    (void))

  (define (cuda-current-context)
    (call-with-foreign-buffer
     'void*
     (lambda (buf)
       (check-cuda cu-ctx-get-current (cu-ctx-get-current buf))
       (foreign-ref 'void* buf 0))))

  (define (cuda-with-context ctx thunk)
    (let ((previous (cuda-current-context)))
      (dynamic-wind
        (lambda ()
          (check-cuda cu-ctx-set-current
                      (cu-ctx-set-current (cuda-context-handle ctx))))
        thunk
        (lambda ()
          (check-cuda cu-ctx-set-current (cu-ctx-set-current previous))))))

  (define (cuda-device-sync!)
    (check-cuda cu-ctx-synchronize (cu-ctx-synchronize))
    (void))

;;; ------------------------------------------------------------------
;;; Device memory
;;; ------------------------------------------------------------------

  (define-record-type (cuda-device-ptr make-cuda-device-ptr cuda-device-ptr?)
    (fields (immutable address) (mutable size)))

  (define (cuda-malloc bytes)
    (call-with-foreign-buffer
     'unsigned-64
     (lambda (buf)
       (check-cuda cu-mem-alloc (cu-mem-alloc buf bytes))
       (make-cuda-device-ptr (foreign-ref 'unsigned-64 buf 0) bytes))))

  (define (cuda-free! p)
    (check-cuda cu-mem-free (cu-mem-free (cuda-device-ptr-address p)))
    (void))

  (define (cuda-memset! p byte count)
    (check-cuda cu-memset-d8!
                (cu-memset-d8! (cuda-device-ptr-address p) byte count))
    (void))

  (define (cuda-memcpy-host->device! p bv)
    (let ((n (bytevector-length bv)))
      (unless (<= n (cuda-device-ptr-size p))
        (assertion-violation 'cuda-memcpy-host->device!
                             "bytevector larger than device buffer" bv))
      (let ((buf (bytevector->foreign-memory bv)))
        (dynamic-wind
          (lambda () #f)
          (lambda ()
            (check-cuda cu-memcpy-host-to-device!
                        (cu-memcpy-host-to-device!
                         (cuda-device-ptr-address p) buf n))
            (void))
          (lambda () (foreign-free buf))))))

  (define (cuda-memcpy-device->host p len)
    (let ((bv (make-bytevector len))
          (buf (foreign-alloc (max len 1))))
      (dynamic-wind
        (lambda () #f)
        (lambda ()
          (check-cuda cu-memcpy-device-to-host!
                      (cu-memcpy-device-to-host!
                       buf (cuda-device-ptr-address p) len))
          (let loop ((i 0))
            (when (< i len)
              (bytevector-u8-set! bv i (foreign-ref 'unsigned-8 buf i))
              (loop (+ i 1))))
          bv)
        (lambda () (foreign-free buf)))))

  (define (cuda-memcpy-device->device! dst src count)
    (check-cuda cu-memcpy-device-to-device!
                (cu-memcpy-device-to-device!
                 (cuda-device-ptr-address dst)
                 (cuda-device-ptr-address src)
                 count))
    (void))

;;; ------------------------------------------------------------------
;;; Modules & functions
;;; ------------------------------------------------------------------

  (define-record-type (cuda-module make-cuda-module cuda-module?)
    (fields (immutable handle)))

  (define-record-type (cuda-function make-cuda-function cuda-function?)
    (fields (immutable handle) (immutable module)))

  ;; IMAGE may be a string (PTX source, loaded via JIT) or a bytevector
  ;; (a cubin/fatbin image).
  (define (cuda-module-load-data image)
    (call-with-foreign-buffer
     'void*
     (lambda (buf)
       (cond ((string? image)
              (check-cuda cu-module-load-data
                          (cu-module-load-data buf image)))
             ((bytevector? image)
              (let ((mem (bytevector->foreign-memory image)))
                (dynamic-wind
                  (lambda () #f)
                  (lambda ()
                    (check-cuda cu-module-load-data
                                (cu-module-load-data buf mem)))
                  (lambda () (foreign-free mem)))))
             (else
              (assertion-violation 'cuda-module-load-data
                                   "expected PTX string or image bytevector"
                                   image)))
       (make-cuda-module (foreign-ref 'void* buf 0)))))

  (define (cuda-module-unload! mod)
    (check-cuda cu-module-unload (cu-module-unload (cuda-module-handle mod)))
    (void))

  (define (cuda-module-get-function mod name)
    (call-with-foreign-buffer
     'void*
     (lambda (buf)
       (check-cuda cu-module-get-function
                   (cu-module-get-function buf (cuda-module-handle mod) name))
       (make-cuda-function (foreign-ref 'void* buf 0) mod))))

;;; ------------------------------------------------------------------
;;; Streams
;;; ------------------------------------------------------------------

  (define-record-type (cuda-stream make-cuda-stream cuda-stream?)
    (fields (immutable handle)))

  ;; The legacy default stream.
  (define cuda-stream-default #f)

  (define (cuda-stream-create)
    (call-with-foreign-buffer
     'void*
     (lambda (buf)
       (check-cuda cu-stream-create (cu-stream-create buf 0))
       (make-cuda-stream (foreign-ref 'void* buf 0)))))

  (define (cuda-stream-destroy! stream)
    (check-cuda cu-stream-destroy (cu-stream-destroy (cuda-stream-handle stream)))
    (void))

  (define (cuda-stream-synchronize stream)
    (check-cuda cu-stream-synchronize
                (cu-stream-synchronize
                 (if stream (cuda-stream-handle stream) 0)))
    (void)))
