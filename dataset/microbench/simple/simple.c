// #include <stdio.h>

// Simple loop with an array
void func(int sum[1]) {
  int i;

#define N 4
  volatile int a[N] = {1, 2, 3, 4};
  volatile int b[N] = {5, 6, 7, 8};
  volatile int c[N] = {0};

  for (i = 0; i < N; i++) {
    // DFGLoop: loop
    c[i] = a[i] + b[i];
  }

  //   int sum = 0;
  for (i = 0; i < N; i++) {
    // printf("c[%d] = %d\n", i, c[i]);
    sum[0] += c[i];
  }

  //   return sum;
}
