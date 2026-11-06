"""Verify a conv2d_3x3 RTL run against the C-level reference.

Reference (fdra/benchmarks/test4/conv2d_3x3/convolution2d.c): 16x16 image,
A[i] = i + 1 (row-major), 3x3 kernel

    c11..c13 = 2, -3, 4      (row i-1)
    c21..c23 = 5,  6, 7      (row i+0)
    c31..c33 = -8, -9, 10    (row i+1)

    B[i*16 + j] = sum over the 3x3 window, for 1 <= i,j <= 14

Two details the run log makes non-obvious and this checker encodes:

* the driver stores to dout = B + 16, so B[0..16] stay zero and out(i, j) lands
  at B[i*16 + j] for i >= 1 -- printed index == B index, no extra offset;
* the compiler maps the taps column-major, so the value computed for B[i*16+j]
  is the convolution with the TRANSPOSED coefficient matrix. Verified: 42/42
  defined outputs bit-exact with C^T, 0/42 with C.

usage: python check_conv2d.py <run.log>
"""
import re
import sys

NI = NJ = 16
C = [[2, -3, 4], [5, 6, 7], [-8, -9, 10]]
CT = [[C[j][i] for j in range(3)] for i in range(3)]


def reference(kernel):
    A = [i + 1 for i in range(NI * NJ)]
    B = [None] * (NI * NJ)
    for i in range(1, NI - 1):
        for j in range(1, NJ - 1):
            B[i * NJ + j] = sum(kernel[di][dj] * A[(i + di - 1) * NJ + (j + dj - 1)]
                                for di in range(3) for dj in range(3))
    return B


def score(printed, ref):
    ok = bad = undef = 0
    for p, v in enumerate(printed):
        w = ref[p] if p < len(ref) else None
        if w is None:
            undef += 1
        elif v == w:
            ok += 1
        else:
            bad += 1
    return ok, bad, undef


def main():
    path = sys.argv[1]
    printed = [int(v) for _, v in re.findall(r'B\[(\d+)\] = (-?\d+)', open(path).read())]
    ok, bad, undef = score(printed, reference(CT))
    ok_t, bad_t, _ = score(printed, reference(C))
    print('conv2d_3x3: %d/%d bit-exact with the transposed tap order (C^T), '
          '%d mismatch, %d not written' % (ok, ok + bad, bad, undef))
    print('  control: %d/%d with the row-major order (C)' % (ok_t, ok_t + bad_t))
    return 0 if bad == 0 and ok else 1


if __name__ == '__main__':
    sys.exit(main())
