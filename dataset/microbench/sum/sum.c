// #include <stdio.h>

volatile int *n = (int *)0;
static int *a = (int *)(1 * sizeof(int));
#define N 100
// Simple accumulation
void func(int sum[1]) {

  //   int N = *n;
  //   int sum = 0;
  int i;
  for (i = 0; i < N; i++) {
    // DFGLoop: loop
    sum[0] += a[i];
  }
  //   printf("sum = %d\n", sum);

  // return sum;
}
