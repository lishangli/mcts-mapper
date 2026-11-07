"""Recover the sample skew between the operands of one node, by brute force.

A linear kernel cannot separate a latency from a skew: both shift every sample by
a constant, so any `(L, s)` pair looks equally good. A kernel whose output is
**quadratic** in the sample index can: the skew changes the linear coefficient
while the quadratic one stays fixed.

Kernels here (ramped input `x = a + j` at sample j, capture index i, j = i - L):

  quad    y = (x + 1) * x   ->  (a+j+1) * (a+j+s)     dfg_quad.json
  square  y = x * x         ->  (a+j)   * (a+j+s)     dfg_square.json

so a grid search over `(L, s)` fits the whole capture, and `s != 0` means the two
operands of the multiply read different samples.

usage: python fit_skew.py <run.log> [--kernel=quad] [--base=1] [--channel=0]
"""
import re
import sys


def model(kernel, a, j, s):
    if kernel == 'square':
        return (a + j) * (a + j + s)
    return (a + j + 1) * (a + j + s)


def main():
    argv = sys.argv[1:]
    base, ch, kernel = 1, 0, 'quad'
    for x in argv:
        if x.startswith('--base='):
            base = int(x.split('=')[1])
        if x.startswith('--channel='):
            ch = int(x.split('=')[1])
        if x.startswith('--kernel='):
            kernel = x.split('=')[1]
    log = [x for x in argv if not x.startswith('--')][0]
    t = open(log).read()
    vals = [int(v) for v in re.findall(r'O%d\[\d+\] = (-?\d+)' % ch, t)]
    start = 0
    for i in range(2, len(vals) - 4):
        if vals[i] > vals[i - 1] and vals[i + 1] > vals[i] and vals[i + 2] > vals[i + 1]:
            start = i
            break
    end = len(vals)
    while end > start and vals[end - 1] == 0:
        end -= 1
    best = None
    for L in range(0, 60):
        for s in range(-4, 5):
            err = 0
            n = 0
            for i in range(start, end):
                e = vals[i] - model(kernel, base, i - L, s)
                err += e * e
                n += 1
            if n and (best is None or err < best[0]):
                best = (err, L, s)
    err, L, s = best
    rmse = (err / max(1, end - start)) ** 0.5
    print('kernel=%s samples=%d (idx %d..%d)' % (kernel, end - start, start, end - 1))
    print('best fit: latency=%d skew=%+d samples, rmse=%.2f' % (L, s, rmse))
    return 0 if abs(s) < 0.5 else 1


if __name__ == '__main__':
    sys.exit(main())
