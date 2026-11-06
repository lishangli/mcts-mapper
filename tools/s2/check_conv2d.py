"""Verify a conv2d_3x3 RTL run against the C-level reference.

Reference (fdra/benchmarks/test4/conv2d_3x3/convolution2d.c): 16x16 image,
A[i] = i + 1 (row-major), 3x3 kernel

    c11..c13 = 2, -3, 4      (row i-1)
    c21..c23 = 5,  6, 7      (row i+0)
    c31..c33 = -8, -9, 10    (row i+1)

    B[i*16 + j] = sum over the 3x3 window, for 1 <= i,j <= 14

The CGRA driver stores to dout = B + 16 (B + 64 bytes), so printed index p
corresponds to B[p + 16].

usage: python check_conv2d.py <run.log> [--offset 16]
"""
import re
import sys

NI = NJ = 16
C = [[2, -3, 4], [5, 6, 7], [-8, -9, 10]]


def reference():
    A = [i + 1 for i in range(NI * NJ)]
    B = [None] * (NI * NJ)
    for i in range(1, NI - 1):
        for j in range(1, NJ - 1):
            B[i * NJ + j] = sum(C[di][dj] * A[(i + di - 1) * NJ + (j + dj - 1)]
                                for di in range(3) for dj in range(3))
    return B


def main():
    path = sys.argv[1]
    off = int(sys.argv[sys.argv.index('--offset') + 1]) if '--offset' in sys.argv else 16
    printed = [int(v) for _, v in re.findall(r'B\[(\d+)\] = (-?\d+)', open(path).read())]
    ref = reference()
    ok = bad = undef = 0
    for p, v in enumerate(printed):
        want = ref[p + off] if p + off < len(ref) else None
        if want is None:
            undef += 1
        elif v == want:
            ok += 1
        else:
            bad += 1
            if bad <= 5:
                print('MISMATCH B[%d] got %d want %d (B[%d])' % (p, v, want, p + off))
    print('conv2d_3x3: %d/%d bit-exact, %d mismatch, %d undefined (out of the '
          'interior)' % (ok, ok + bad, bad, undef))
    return 0 if bad == 0 and ok else 1


if __name__ == '__main__':
    sys.exit(main())
