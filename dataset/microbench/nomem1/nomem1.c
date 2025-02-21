// #include <stdio.h>

static volatile int *n;
static int count[1];

void func(int *sum) {

  int N = 100;
  //   int sum = 0;
  int i;
  //   int count[1] = {0};
  for (i = 1; i <= N; i++) {
    // DFGLoop: loop
    sum[0] += count[0] * 3;
    count[0] += 1;
  }
  //   printf("sum = %d\n", sum);

  //   return sum;
}
