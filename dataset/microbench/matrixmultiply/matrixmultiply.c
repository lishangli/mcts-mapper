// matrix multiply
// Author: Andrew Canis
// Date: June 13, 2012

// #include <stdio.h>

// matrices are SIZE x SIZE
#define SIZE 20

static int (*A1)[SIZE] = (int (*)[SIZE])0xA00;
static int (*B1)[SIZE] = (int (*)[SIZE])0xB00;

volatile int resultAB1[SIZE][SIZE];

__attribute__((noinline)) void multiply(int i, int j, int sum[1]) {
  int k = 0;
  for (k = 0; k < SIZE; k++) {
    // DFGLoop: loop
    sum[0] += A1[i][k] * B1[k][j];
  }
  resultAB1[i][j] = sum[0];
  //   return sum;
}

void func(int count[1]) {
  int i, j;

  // unsigned long long count[1] = {0};
  for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
      // multiply(i, j, count);
      for (int k = 0; k < SIZE; k++) {
        // DFGLoop: loop
        count[0] += A1[i][k] * B1[k][j];
        resultAB1[i][j] = count[0];
      }
    }
  }

  //   printf("Result: %lld\n", count);
  //   if (count == 962122000) {
  //     printf("RESULT: PASS\n");
  //   } else {
  //     printf("RESULT: FAIL\n");
  //   }
  //   return count;
}
