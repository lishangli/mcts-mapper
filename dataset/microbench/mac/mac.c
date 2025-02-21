// #include <stdio.h>

volatile int *N = (int *)0;
static int *a = (int *)(1 * sizeof(int));
static int *b = (int *)(9 * sizeof(int));

void func(int *sum) {
  int i;
  int n = 100;

  //   int sum = 0;

  for (i = 1; i < n - 1; i++) {
    // DFGLoop: loop
    sum[0] += a[i] * b[i];
  }

  //   return sum;
}
