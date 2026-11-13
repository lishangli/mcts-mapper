#include <stdio.h>
#include <stdint.h>
extern void cgra_execute0(void** din_addr, void** dout_addr);
static inline unsigned long cyc(void) { unsigned long c;
  __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }

#define N 32
static int m1[N][N] __attribute__((aligned(64)));
static int m2[N][N] __attribute__((aligned(64)));
static int prod[N][N] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < N; i++)
    for (int k = 0; k < N; k++) {
      m1[i][k] = (i % 5) - 2;
      m2[i][k] = (k % 7) - 3;
    }
  __asm__ volatile("fence" ::: "memory");
  void* din[8];
  for (int k = 0; k < 4; k++) din[k] = (void*)m1;
  for (int k = 4; k < 8; k++) din[k] = (void*)m2;
  void* dout[1]; dout[0] = (void*)prod;
  unsigned long t0 = cyc();
  cgra_execute0(din, dout);
  unsigned long t1 = cyc();
  int bad = 0; long ref0 = 0;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      long s = 0;
      for (int k = 0; k < N; k++) s += (long)m1[i][k] * (long)m2[k][j];
      if (i == 0 && j == 0) ref0 = s;
      if (prod[i][j] != (int)s) bad++;
    }
  printf("GEMM_PERF cycles=%lu ops=%d bad=%d ref00=%ld got00=%d\n",
         t1 - t0, 2*N*N*N, bad, ref0, prod[0][0]);
  return 0;
}
