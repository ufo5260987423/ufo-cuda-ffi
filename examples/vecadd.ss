#!r6rs ; -*- mode: scheme; coding: utf-8 -*-
;;; examples/vecadd.ss — Vector addition on the GPU (Tier-3 example).
;;;
;;; This exercises the full pipeline and therefore needs a machine with an
;;; NVIDIA GPU and the driver installed.  On a GPU-less machine the
;;; (cuda-init!) call raises &cuda-error with code CUDA_ERROR_STUB_LIBRARY
;;; (34, linked against the toolkit stub) or CUDA_ERROR_NO_DEVICE (100).
;;;
;;; Compile the kernel first (from the project root, inside `nix develop`):
;;;   nvcc -cubin -gencode=arch=compute_XX,code=sm_XX -o examples/vecadd.cubin examples/vecadd.cu
;;; or let this script do it via (cuda-compile-cubin ...).
;;;
;;; Run:  scheme --libdirs . --script examples/vecadd.ss

(import (chezscheme)
        (ufo cuda)
        (ufo cuda kernel))

(define n 1024)
(define n-bytes (* n 4))

(define (read-binary-file path)
  (call-with-port (open-file-input-port path)
    (lambda (port)
      (let loop ((acc '()))
        (let ((b (get-u8 port)))
          (if (eof-object? b)
              (let* ((len (length acc))
                     (bv (make-bytevector len)))
                (let fill ((i 0) (rest (reverse acc)))
                  (unless (null? rest)
                    (bytevector-u8-set! bv i (car rest))
                    (fill (+ i 1) (cdr rest))))
                bv)
              (loop (cons b acc))))))))

(define (make-float-vector k fill)
  (let ((bv (make-bytevector n-bytes)))
    (do ((i 0 (+ i 1)))
        ((= i k) bv)
      (bytevector-ieee-single-native-set! bv (* i 4) (fill i)))))

;; -------------------------------------------------------------------

(cuda-load!)
(cuda-init!)

(define device (cuda-get-device 0))
(printf "device 0: ~a (compute capability ~a.~a)~%"
        (cuda-device-name device)
        (car (cuda-device-compute-capability device))
        (cdr (cuda-device-compute-capability device)))

(define context (cuda-primary-context device))

(define cubin-path "examples/vecadd.cubin")
(unless (file-exists? cubin-path)
  (printf "compiling kernel with nvcc...~%")
  (cuda-compile-cubin "examples/vecadd.cu" cubin-path
                      (cuda-device-gencode device)))

(cuda-with-context
 context
 (lambda ()
   (define module (cuda-module-load-data (read-binary-file cubin-path)))
   (define vecadd (cuda-module-get-function module "vecadd"))

   (define a-host (make-float-vector n (lambda (i) (exact->inexact i))))
   (define b-host (make-float-vector n (lambda (i) (exact->inexact (* 2 i)))))

   (define a-dev (cuda-malloc n-bytes))
   (define b-dev (cuda-malloc n-bytes))
   (define c-dev (cuda-malloc n-bytes))

   (cuda-memcpy-host->device! a-dev a-host)
   (cuda-memcpy-host->device! b-dev b-host)

   (define threads 256)
   (define blocks (div (+ n threads -1) threads))
   (cuda-launch! vecadd (list blocks 1 1) (list threads 1 1)
                 (list a-dev b-dev c-dev (cons 'int n)))
   (cuda-device-sync!)

   (let* ((c-host (cuda-memcpy-device->host c-dev n-bytes))
          (ok #t))
     (do ((i 0 (+ i 1)))
         ((or (= i n) (not ok)))
       (let ((got (bytevector-ieee-single-native-ref c-host (* i 4)))
             (want (exact->inexact (* 3 i))))
         (unless (= got want)
           (printf "mismatch at ~a: got ~a, want ~a~%" i got want)
           (set! ok #f))))
     (printf "~a~%" (if ok "PASS: c[i] == 3*i for all i" "FAIL")))

   (cuda-free! a-dev)
   (cuda-free! b-dev)
   (cuda-free! c-dev)
   (cuda-module-unload! module)))
