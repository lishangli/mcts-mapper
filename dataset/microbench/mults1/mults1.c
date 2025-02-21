// #include <stdio.h>

volatile int *N = (int *)0xf00;
static int *a = (int *)0xa00;
static int *b = (int *)0xb00;
static int *m = (int *)0xe00;

void func(int sum[1]) {
  int i, j;
  int n = 100;

  for (i = 1; i < n - 1; i++) {
    // DFGLoop: loop
    sum[0] += a[i] * 10 + a[i + 1] * 20 + a[i + 2] * 39 + a[i + 3] * 15;
  }

  //   return sum;
}
