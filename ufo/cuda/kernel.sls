#!r6rs ; -*- mode: scheme; coding: utf-8 -*-
;;; ufo/cuda/kernel.sls — Kernel launching and nvcc compilation helpers.
;;;
;;; cuda-launch! marshals Scheme values into the void** kernelParams array
;;; that cuLaunchKernel expects.  Each element of PARAMS is either
;;;   - a (ftype . value) pair, e.g. ((unsigned-64 . n) (float . 2.5)), or
;;;   - a bare <cuda-device-ptr>, passed as its device address.
(library (ufo cuda kernel)
  (export cuda-launch! cuda-nvcc cuda-compile-cubin cuda-device-gencode)

  (import (chezscheme)
          (ufo cuda)
          (ufo cuda ffi))

  ;; cuda-check is exported by (ufo cuda); it raises &cuda-error on
  ;; non-success CUresult.  (It is a macro, so it re-expands here.)

  (define (single-or-triple x who)
    (cond ((and (integer? x) (positive? x)) (values x 1 1))
          ((and (list? x) (<= 1 (length x) 3)
                (for-all (lambda (n) (and (integer? n) (positive? n))) x))
           (values (list-ref x 0)
                   (if (> (length x) 1) (list-ref x 1) 1)
                   (if (> (length x) 2) (list-ref x 2) 1)))
          (else
           (assertion-violation who
             "expected a positive integer or a list of 1..3 integers" x))))

  (define (normalize-param p)
    (cond ((cuda-device-ptr? p)
           (cons 'unsigned-64 (cuda-device-ptr-address p)))
          ((and (pair? p) (symbol? (car p)))
           (let ((v (cdr p)))
             (cons (car p)
                   (if (cuda-device-ptr? v) (cuda-device-ptr-address v) v))))
          (else
           (assertion-violation 'cuda-launch!
                                "bad kernel parameter (want (type . value) or <cuda-device-ptr>)"
                                p))))

  (define cuda-launch!
    (case-lambda
      [(func grid block params)
       (cuda-launch! func grid block params 0 #f)]
      [(func grid block params shared-mem-bytes)
       (cuda-launch! func grid block params shared-mem-bytes #f)]
      [(func grid block params shared-mem-bytes stream)
       (let-values ([(gx gy gz) (single-or-triple grid 'cuda-launch!)]
                    [(bx by bz) (single-or-triple block 'cuda-launch!)])
         (let* ((params (map normalize-param params))
                (n (length params))
                (argv (foreign-alloc (* (max n 1) (foreign-sizeof 'void*))))
                (bufs '()))
           (dynamic-wind
             (lambda () #f)
             (lambda ()
               (for-each
                (lambda (param i)
                  (let* ((type (car param))
                         (value (cdr param))
                         (buf (foreign-alloc (foreign-sizeof type))))
                    (set! bufs (cons buf bufs))
                    (foreign-set! type buf 0 value)
                    (foreign-set! 'void* argv (* i (foreign-sizeof 'void*)) buf)))
                params
                (iota n))
               (cuda-check cu-launch-kernel
                 (cu-launch-kernel (cuda-function-handle func)
                                   gx gy gz bx by bz
                                   shared-mem-bytes
                                   (if stream (cuda-stream-handle stream) 0)
                                   argv 0))
               (void))
             (lambda ()
               (for-each foreign-free bufs)
               (foreign-free argv)))))]))

  ;; ----------------------------------------------------------------
  ;; nvcc compilation
  ;; ----------------------------------------------------------------

  (define cuda-nvcc (make-parameter "nvcc"))

  ;; "-gencode=arch=compute_XY,code=sm_XY" for a device's capability.
  (define (cuda-device-gencode dev)
    (let* ((cap (cuda-device-compute-capability dev))
           (xy (string-append (number->string (car cap))
                              (number->string (cdr cap)))))
      (string-append "-gencode=arch=compute_" xy ",code=sm_" xy)))

  ;; Compile SOURCE-PATH to a cubin at OUTPUT-PATH with nvcc.
  ;; Extra command-line flags may be passed as strings, e.g.
  ;;   (cuda-compile-cubin "vecadd.cu" "vecadd.cubin"
  ;;                       (cuda-device-gencode dev) "-O3")
  (define cuda-compile-cubin
    (case-lambda
      [(source-path output-path)
       (cuda-compile-cubin source-path output-path '())]
      [(source-path output-path flags)
       (let ((cmd (string-append
                   (cuda-nvcc) " -cubin"
                   (apply string-append
                          (map (lambda (f) (string-append " " f)) flags))
                   " -o " output-path " " source-path)))
         (unless (zero? (system cmd))
           (assertion-violation 'cuda-compile-cubin
                                "nvcc failed" cmd))
         (void))])))
