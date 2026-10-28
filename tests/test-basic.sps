#!r6rs ; -*- mode: scheme; coding: utf-8 -*-
;;; tests/test-basic.sps — Tier-1 tests: everything that can be verified
;;; without a GPU, using the CUDA driver stub (every entry point returns
;;; CUDA_ERROR_STUB_LIBRARY = 34).
;;;
;;; Run:  scheme --libdirs . --script tests/test-basic.sps [libcuda-path]
;;; The path defaults to $UFO_CUDA_LIB (set by `nix develop`).

(import (chezscheme)
        (ufo cuda)
        (ufo cuda ffi)
        (ufo cuda kernel))

(define checks 0)
(define failures 0)

(define (report-ok desc)
  (set! checks (+ checks 1))
  (printf "ok ~a\n" desc))

(define (report-fail desc . info)
  (set! checks (+ checks 1))
  (set! failures (+ failures 1))
  (apply fprintf (cons (current-error-port)
                       (cons (string-append "FAIL " desc "\n") info))))

(define (check desc pred)
  (if pred (report-ok desc) (report-fail desc)))

(define (check-equal desc a b)
  (if (equal? a b)
      (report-ok desc)
      (report-fail desc "  expected ~s, got ~s\n" b a)))

(define (string-contains str sub)
  (let ((n (string-length str)) (m (string-length sub)))
    (let loop ((i 0))
      (cond ((> (+ i m) n) #f)
            ((string=? (substring str i (+ i m)) sub) #t)
            (else (loop (+ i 1)))))))

;; Evaluate THUNK; return the raised exception object, or #f if it
;; returned normally.
(define (caught-exception thunk)
  (call/cc
   (lambda (k)
     (with-exception-handler
      (lambda (e) (k e))
      (lambda () (thunk) #f)))))

;;; ------------------------------------------------------------------

(define stub-path
  (cond ((and (> (length (command-line)) 1) (cadr (command-line))))
        ((getenv "UFO_CUDA_LIB"))
        (else
         (fprintf (current-error-port)
                  "usage: test-basic.sps [libcuda-path]  (or set $UFO_CUDA_LIB)\n")
         (exit 2))))

(printf ";; using driver library: ~a\n" stub-path)

;; --- driver loading -------------------------------------------------

(cuda-load! stub-path)
(check "cuda-load! loads the library" (cuda-loaded?))
(check "cuda-init is bound" (procedure? cu-init))
(check "cu-launch-kernel is bound" (procedure? cu-launch-kernel))
(check "cu-mem-alloc (v2) is bound" (procedure? cu-mem-alloc))

;; Calling an API before initialization must raise a Scheme error
;; (assertion), not crash.
(check "init required before use"
       (let ((e (caught-exception (lambda () (cuda-device-count)))))
         (and (not (cuda-error? e)) (condition? e))))

;; --- initialization through the stub --------------------------------

;; The stub always answers CUDA_ERROR_STUB_LIBRARY (34).  From the
;; library's point of view this exercises the same code path as any
;; other failed driver call.
(let ((e (caught-exception (lambda () (cuda-init!)))))
  (check "cuda-init! raises &cuda-error on stub"
         (and (cuda-error? e) (eq? (cuda-error-code e) 34)))
  (check "cuda-error carries api name"
         (and (cuda-error? e) (symbol? (cuda-error-api e))))
  (check "cuda-error message mentions STUB_LIBRARY"
         (and (cuda-error? e)
              (string-contains (condition-message e)
                               "CUDA_ERROR_STUB_LIBRARY"))))

;; A failed init must not mark the driver as initialized.
(check "failed init leaves uninitialized state" (not (cuda-initialized?)))

;; --- error table -----------------------------------------------------

(check-equal "code 34 name" (cuda-error-code->symbol 34) 'CUDA_ERROR_STUB_LIBRARY)
(check-equal "code 0 name" (cuda-error-code->symbol 0) 'CUDA_SUCCESS)
(check-equal "code 100 name" (cuda-error-code->symbol 100) 'CUDA_ERROR_NO_DEVICE)
(check-equal "code 999 name" (cuda-error-code->symbol 999) 'CUDA_ERROR_UNKNOWN)
(check-equal "code -1 name" (cuda-error-code->symbol -1) 'CUDA_ERROR_UNKNOWN)
(check-equal "symbol->code roundtrip"
             (cuda-error-symbol->code 'CUDA_ERROR_NO_DEVICE) 100)
(check-equal "unknown symbol"
             (cuda-error-symbol->code 'CUDA_ERROR_NOT_A_REAL_CODE) #f)

;; --- summary ---------------------------------------------------------

(printf ";; ~a checks, ~a failure(s)\n" checks failures)
(exit (if (zero? failures) 0 1))
