#include <stdio.h>
#include <stdint.h>
extern void cgra_execute0(void** din_addr, void** dout_addr);
static inline unsigned long cyc(void) { unsigned long c;
  __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }
int A[256] __attribute__((aligned(64)));
int B[256] __attribute__((aligned(64)));
int main(void) {
  for (int i = 0; i < 256; i++) A[i] = i + 1;
  __asm__ volatile("fence" ::: "memory");   /* let the stores retire before DMA */
  void* din[9]; void* dout[1];
  for (int k = 0; k < 9; k++) din[k] = (void*)A;
  dout[0] = (void*)(B + 16);
  unsigned long t0 = cyc();
  cgra_execute0(din, dout);
  unsigned long t1 = cyc();
  printf("CONV_PERF cycles=%lu\n", t1 - t0);
  printf("conv out:\n");
  for (int i = 0; i < 64; i++) printf("B[%d] = %d\n", i, B[i]);
  return 0;
}
