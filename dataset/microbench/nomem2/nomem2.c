// #include <stdio.h>

static volatile int n = 10;
static int count[1];
void func(int sum[1]) {

  int N = 100;

  int i;
  int j = 0;
  int k = 0;
  for (i = 0; i < N; i++) {
    // DFGLoop: loop
    sum[0] += k * 3 + 5;
    k += 1;
  }
  //   printf("sum = %d\n", sum);

  //   return sum;
}
