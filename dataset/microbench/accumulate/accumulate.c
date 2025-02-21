// #include <stdio.h>

volatile int *N = (int *)0xf00;
static int *a = (int *)0xa00;
static int *b = (int *)0xb00;
static int *c = (int *)0xc00;
// static int *sum = (int *)0xd00;

void func(int *sum) {
  int i;
  int n = 100;

  // int sum = 0;
  for (i = 1; i < n - 1; i++) {
    // DFGLoop: loop
    c[i] *= a[i + 1] + b[i - 1];
    sum[0] += c[i];
  }

  // return sum[0];
}
