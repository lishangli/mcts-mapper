// #include <stdio.h>

volatile int *N = (int *)0xf00;
static int *a = (int *)0xa00;
static int *b = (int *)0xb00;
static int *c = (int *)0xc00;
static int *d = (int *)0xd00;

void func(int *sum, int *sum2) {
  int i;
  int n = 100;

  // int sum = 0;
  // int sum2 = 0;

  for (i = 1; i < n - 1; i++) {
    // DFGLoop: loop
    sum[0] += a[i] * b[i];
    sum2[0] += a[i] * (b[i] + 1) * c[i] * d[i];
  }

  // return sum + sum2;
}
