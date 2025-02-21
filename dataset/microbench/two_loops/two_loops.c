// #include <stdio.h>

void func(int sum1[1], int sum[1]) {
  // #define N 100
  // #define SQRT 10
#define N 9
#define SQRT 3

  int A[N], B[N];
  int C[N];
  int i, j;
  int tmp;

  //   for (i = 0; i < N; i++) {
  //     A[i] = i;
  //   }

  //   int sum1 = 0;
  for (i = 0; i < N; i++) {
    // DFGLoop: loop1
    tmp = 2 + A[i];
    tmp = tmp * tmp + tmp - 1;
    B[i] = tmp * 3 + 1;
    sum1[0] += B[i];
    //        printf("sum %d = %d\n", i, sum1);
  }

  //   long long sum = 0;
  for (i = 0; i < SQRT * SQRT; i++) {

    // DFGLoop: loop2
    // int idx = i * SQRT + j;
    tmp = 2 + B[i];
    tmp = tmp * tmp * tmp + tmp * tmp + 10;
    C[i] = (tmp + 10) * 2;
    //            printf("C[%d] = %lld\n", idx, C[idx]);
    sum[0] += C[i];
    //            printf("sum %d = %lld\n", idx, sum);
  }

  //   printf("sum = %lld\n", sum);

  //   return sum;
}
