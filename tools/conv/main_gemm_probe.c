#include <stdio.h>
extern void cgra_execute0(void** din_addr, void** dout_addr);
static inline unsigned long cyc(void) { unsigned long c;
  __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }
#define N 32
static int m1[N][N] __attribute__((aligned(64)));
static int m2[N][N] __attribute__((aligned(64)));
static int prod[N][N] __attribute__((aligned(64)));
int main(void) {
  for (int i = 0; i < N; i++)
    for (int k = 0; k < N; k++) { m1[i][k] = (i % 5) - 2; m2[i][k] = (k % 7) - 3; }
  __asm__ volatile("fence" ::: "memory");
  void* din[8];
  for (int k = 0; k < 4; k++) din[k] = (void*)m1;
  for (int k = 4; k < 8; k++) din[k] = (void*)m2;
  void* dout[1]; dout[0] = (void*)prod;
  unsigned long t0 = cyc();
  cgra_execute0(din, dout);
  unsigned long t1 = cyc();
  int h8 = 0, h32 = 0;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      int s8 = 0; long s32 = 0;
      for (int k = 0; k < 8; k++) s8 += m1[i][k] * m2[k][j];
      for (int k = 0; k < N; k++) s32 += (long)m1[i][k] * (long)m2[k][j];
      if (prod[i][j] == s8) h8++;
      if (prod[i][j] == (int)s32) h32++;
    }
  printf("GEMM_PROBE cycles=%lu hit8=%d hit32=%d p00=%d\n", t1 - t0, h8, h32, prod[0][0]);
  return 0;
}
