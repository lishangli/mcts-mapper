void cgra_execute(void** din_addr, void** dout_addr)
{
	 #ifdef CGRA_DEBUG
	 printf("|");
	 #endif
	static unsigned short cin[0][3] __attribute__((aligned(16))) = {
	};

	load_cfg((void*)cin, 0x100000, 0, 0, 0);
	 int sizes_2[1] = {512};
	 int strides_2[1] = {1};
	load_mem(din_addr[0] + 0, 0x80000, 0, 0, 0, strides_2, sizes_2, 1, 0);
	 int sizes_3[1] = {512};
	 int strides_3[1] = {1};
	load_mem(din_addr[1] + 0, 0x0, 0, 0, 0, strides_3, sizes_3, 1, 0);
	 int sizes_4[1] = {512};
	 int strides_4[1] = {1};
	load_mem(din_addr[2] + 0, 0x8000, 0, 0, 0, strides_4, sizes_4, 1, 0);
	 int sizes_5[1] = {512};
	 int strides_5[1] = {1};
	load_mem(din_addr[3] + 0, 0x10000, 0, 0, 0, strides_5, sizes_5, 1, 0);
	config(0x0, 0, 0, 0);
	execute(0x4011c, 0, 0);
	 int sizes_10[1] = {512};
	 int strides_10[1] = {1};
	store_mem(dout_addr[0] + 0, 0x18000, 0, 0, strides_10, sizes_10, 1, 0);
fence(1);}
void cgra_execute(void** din_addr, void** dout_addr)
{
	 #ifdef CGRA_DEBUG
	 printf("|");
	 #endif
	static unsigned short cin[0][3] __attribute__((aligned(16))) = {
	};

	load_cfg((void*)cin, 0x100000, 0, 0, 0);
	 int sizes_2[1] = {512};
	 int strides_2[1] = {1};
	load_mem(din_addr[0] + 0, 0x80000, 0, 0, 0, strides_2, sizes_2, 1, 0);
	 int sizes_3[1] = {512};
	 int strides_3[1] = {1};
	load_mem(din_addr[1] + 0, 0x0, 0, 0, 0, strides_3, sizes_3, 1, 0);
	 int sizes_4[1] = {512};
	 int strides_4[1] = {1};
	load_mem(din_addr[2] + 0, 0x8000, 0, 0, 0, strides_4, sizes_4, 1, 0);
	 int sizes_5[1] = {512};
	 int strides_5[1] = {1};
	load_mem(din_addr[3] + 0, 0x10000, 0, 0, 0, strides_5, sizes_5, 1, 0);
	config(0x0, 0, 0, 0);
	execute(0x4011c, 0, 0);
	 int sizes_10[1] = {512};
	 int strides_10[1] = {1};
	store_mem(dout_addr[0] + 0, 0x18000, 0, 0, strides_10, sizes_10, 1, 0);
fence(1);}
void cgra_execute(void** din_addr, void** dout_addr)
{
	 #ifdef CGRA_DEBUG
	 printf("|");
	 #endif
	static unsigned short cin[0][3] __attribute__((aligned(16))) = {
	};

	load_cfg((void*)cin, 0x100000, 0, 0, 0);
	 int sizes_2[1] = {512};
	 int strides_2[1] = {1};
	load_mem(din_addr[0] + 0, 0x80000, 0, 0, 0, strides_2, sizes_2, 1, 0);
	 int sizes_3[1] = {512};
	 int strides_3[1] = {1};
	load_mem(din_addr[1] + 0, 0x0, 0, 0, 0, strides_3, sizes_3, 1, 0);
	 int sizes_4[1] = {512};
	 int strides_4[1] = {1};
	load_mem(din_addr[2] + 0, 0x8000, 0, 0, 0, strides_4, sizes_4, 1, 0);
	 int sizes_5[1] = {512};
	 int strides_5[1] = {1};
	load_mem(din_addr[3] + 0, 0x10000, 0, 0, 0, strides_5, sizes_5, 1, 0);
	config(0x0, 0, 0, 0);
	execute(0x4011c, 0, 0);
	 int sizes_10[1] = {512};
	 int strides_10[1] = {1};
	store_mem(dout_addr[0] + 0, 0x18000, 0, 0, strides_10, sizes_10, 1, 0);
fence(1);}
void cgra_execute(void** din_addr, void** dout_addr)
{
	 #ifdef CGRA_DEBUG
	 printf("|");
	 #endif
	static unsigned short cin[0][3] __attribute__((aligned(16))) = {
	};

	load_cfg((void*)cin, 0x100000, 0, 0, 0);
	 int sizes_2[1] = {512};
	 int strides_2[1] = {1};
	load_mem(din_addr[0] + 0, 0x80000, 0, 0, 0, strides_2, sizes_2, 1, 0);
	 int sizes_3[1] = {512};
	 int strides_3[1] = {1};
	load_mem(din_addr[1] + 0, 0x0, 0, 0, 0, strides_3, sizes_3, 1, 0);
	 int sizes_4[1] = {512};
	 int strides_4[1] = {1};
	load_mem(din_addr[2] + 0, 0x8000, 0, 0, 0, strides_4, sizes_4, 1, 0);
	 int sizes_5[1] = {512};
	 int strides_5[1] = {1};
	load_mem(din_addr[3] + 0, 0x10000, 0, 0, 0, strides_5, sizes_5, 1, 0);
	config(0x0, 0, 0, 0);
	execute(0x4011c, 0, 0);
	 int sizes_10[1] = {512};
	 int strides_10[1] = {1};
	store_mem(dout_addr[0] + 0, 0x18000, 0, 0, strides_10, sizes_10, 1, 0);
fence(1);}
void cgra_execute(void** din_addr, void** dout_addr)
{
	 #ifdef CGRA_DEBUG
	 printf("|");
	 #endif
	static unsigned short cin[0][3] __attribute__((aligned(16))) = {
	};

	load_cfg((void*)cin, 0x100000, 0, 0, 0);
	 int sizes_2[1] = {512};
	 int strides_2[1] = {1};
	load_mem(din_addr[0] + 0, 0x80000, 0, 0, 0, strides_2, sizes_2, 1, 0);
	 int sizes_3[1] = {512};
	 int strides_3[1] = {1};
	load_mem(din_addr[1] + 0, 0x0, 0, 0, 0, strides_3, sizes_3, 1, 0);
	 int sizes_4[1] = {512};
	 int strides_4[1] = {1};
	load_mem(din_addr[2] + 0, 0x8000, 0, 0, 0, strides_4, sizes_4, 1, 0);
	 int sizes_5[1] = {512};
	 int strides_5[1] = {1};
	load_mem(din_addr[3] + 0, 0x10000, 0, 0, 0, strides_5, sizes_5, 1, 0);
	config(0x0, 0, 0, 0);
	execute(0x4011c, 0, 0);
	 int sizes_10[1] = {512};
	 int strides_10[1] = {1};
	store_mem(dout_addr[0] + 0, 0x18000, 0, 0, strides_10, sizes_10, 1, 0);
fence(1);}
void cgra_execute(void** din_addr, void** dout_addr)
{
	 #ifdef CGRA_DEBUG
	 printf("|");
	 #endif
	static unsigned short cin[0][3] __attribute__((aligned(16))) = {
	};

	load_cfg((void*)cin, 0x100000, 0, 0, 0);
	 int sizes_2[1] = {512};
	 int strides_2[1] = {1};
	load_mem(din_addr[0] + 0, 0x80000, 0, 0, 0, strides_2, sizes_2, 1, 0);
	 int sizes_3[1] = {512};
	 int strides_3[1] = {1};
	load_mem(din_addr[1] + 0, 0x0, 0, 0, 0, strides_3, sizes_3, 1, 0);
	 int sizes_4[1] = {512};
	 int strides_4[1] = {1};
	load_mem(din_addr[2] + 0, 0x8000, 0, 0, 0, strides_4, sizes_4, 1, 0);
	 int sizes_5[1] = {512};
	 int strides_5[1] = {1};
	load_mem(din_addr[3] + 0, 0x10000, 0, 0, 0, strides_5, sizes_5, 1, 0);
	config(0x0, 0, 0, 0);
	execute(0x4011c, 0, 0);
	 int sizes_10[1] = {512};
	 int strides_10[1] = {1};
	store_mem(dout_addr[0] + 0, 0x18000, 0, 0, strides_10, sizes_10, 1, 0);
fence(1);}
void cgra_execute(void** din_addr, void** dout_addr)
{
	 #ifdef CGRA_DEBUG
	 printf("|");
	 #endif
	static unsigned short cin[0][3] __attribute__((aligned(16))) = {
	};

	load_cfg((void*)cin, 0x100000, 0, 0, 0);
	 int sizes_2[1] = {512};
	 int strides_2[1] = {1};
	load_mem(din_addr[0] + 0, 0x80000, 0, 0, 0, strides_2, sizes_2, 1, 0);
	 int sizes_3[1] = {512};
	 int strides_3[1] = {1};
	load_mem(din_addr[1] + 0, 0x0, 0, 0, 0, strides_3, sizes_3, 1, 0);
	 int sizes_4[1] = {512};
	 int strides_4[1] = {1};
	load_mem(din_addr[2] + 0, 0x8000, 0, 0, 0, strides_4, sizes_4, 1, 0);
	 int sizes_5[1] = {512};
	 int strides_5[1] = {1};
	load_mem(din_addr[3] + 0, 0x10000, 0, 0, 0, strides_5, sizes_5, 1, 0);
	config(0x0, 0, 0, 0);
	execute(0x4011c, 0, 0);
	 int sizes_10[1] = {512};
	 int strides_10[1] = {1};
	store_mem(dout_addr[0] + 0, 0x18000, 0, 0, strides_10, sizes_10, 1, 0);
fence(1);}
void cgra_execute(void** din_addr, void** dout_addr)
{
	 #ifdef CGRA_DEBUG
	 printf("|");
	 #endif
	static unsigned short cin[0][3] __attribute__((aligned(16))) = {
	};

	load_cfg((void*)cin, 0x100000, 0, 0, 0);
	 int sizes_2[1] = {512};
	 int strides_2[1] = {1};
	load_mem(din_addr[0] + 0, 0x80000, 0, 0, 0, strides_2, sizes_2, 1, 0);
	 int sizes_3[1] = {512};
	 int strides_3[1] = {1};
	load_mem(din_addr[1] + 0, 0x0, 0, 0, 0, strides_3, sizes_3, 1, 0);
	 int sizes_4[1] = {512};
	 int strides_4[1] = {1};
	load_mem(din_addr[2] + 0, 0x8000, 0, 0, 0, strides_4, sizes_4, 1, 0);
	 int sizes_5[1] = {512};
	 int strides_5[1] = {1};
	load_mem(din_addr[3] + 0, 0x10000, 0, 0, 0, strides_5, sizes_5, 1, 0);
	config(0x0, 0, 0, 0);
	execute(0x4011c, 0, 0);
	 int sizes_10[1] = {512};
	 int strides_10[1] = {1};
	store_mem(dout_addr[0] + 0, 0x18000, 0, 0, strides_10, sizes_10, 1, 0);
fence(1);}
void cgra_execute(void** din_addr, void** dout_addr)
{
	 #ifdef CGRA_DEBUG
	 printf("|");
	 #endif
	static unsigned short cin[0][3] __attribute__((aligned(16))) = {
	};

	load_cfg((void*)cin, 0x100000, 0, 0, 0);
	 int sizes_2[1] = {512};
	 int strides_2[1] = {1};
	load_mem(din_addr[0] + 0, 0x80000, 0, 0, 0, strides_2, sizes_2, 1, 0);
	 int sizes_3[1] = {512};
	 int strides_3[1] = {1};
	load_mem(din_addr[1] + 0, 0x0, 0, 0, 0, strides_3, sizes_3, 1, 0);
	 int sizes_4[1] = {512};
	 int strides_4[1] = {1};
	load_mem(din_addr[2] + 0, 0x8000, 0, 0, 0, strides_4, sizes_4, 1, 0);
	 int sizes_5[1] = {512};
	 int strides_5[1] = {1};
	load_mem(din_addr[3] + 0, 0x10000, 0, 0, 0, strides_5, sizes_5, 1, 0);
	config(0x0, 0, 0, 0);
	execute(0x4011c, 0, 0);
	 int sizes_10[1] = {512};
	 int strides_10[1] = {1};
	store_mem(dout_addr[0] + 0, 0x18000, 0, 0, strides_10, sizes_10, 1, 0);
fence(1);}
void cgra_execute(void** din_addr, void** dout_addr)
{
	 #ifdef CGRA_DEBUG
	 printf("|");
	 #endif
	static unsigned short cin[0][3] __attribute__((aligned(16))) = {
	};

	load_cfg((void*)cin, 0x100000, 0, 0, 0);
	 int sizes_2[1] = {512};
	 int strides_2[1] = {1};
	load_mem(din_addr[0] + 0, 0x80000, 0, 0, 0, strides_2, sizes_2, 1, 0);
	 int sizes_3[1] = {512};
	 int strides_3[1] = {1};
	load_mem(din_addr[1] + 0, 0x0, 0, 0, 0, strides_3, sizes_3, 1, 0);
	 int sizes_4[1] = {512};
	 int strides_4[1] = {1};
	load_mem(din_addr[2] + 0, 0x8000, 0, 0, 0, strides_4, sizes_4, 1, 0);
	 int sizes_5[1] = {512};
	 int strides_5[1] = {1};
	load_mem(din_addr[3] + 0, 0x10000, 0, 0, 0, strides_5, sizes_5, 1, 0);
	config(0x0, 0, 0, 0);
	execute(0x4011c, 0, 0);
	 int sizes_10[1] = {512};
	 int strides_10[1] = {1};
	store_mem(dout_addr[0] + 0, 0x18000, 0, 0, strides_10, sizes_10, 1, 0);
fence(1);}
void cgra_execute(void** din_addr, void** dout_addr)
{
	 #ifdef CGRA_DEBUG
	 printf("|");
	 #endif
	static unsigned short cin[0][3] __attribute__((aligned(16))) = {
	};

	load_cfg((void*)cin, 0x100000, 0, 0, 0);
	 int sizes_2[1] = {512};
	 int strides_2[1] = {1};
	load_mem(din_addr[0] + 0, 0x80000, 0, 0, 0, strides_2, sizes_2, 1, 0);
	 int sizes_3[1] = {512};
	 int strides_3[1] = {1};
	load_mem(din_addr[1] + 0, 0x0, 0, 0, 0, strides_3, sizes_3, 1, 0);
	 int sizes_4[1] = {512};
	 int strides_4[1] = {1};
	load_mem(din_addr[2] + 0, 0x8000, 0, 0, 0, strides_4, sizes_4, 1, 0);
	 int sizes_5[1] = {512};
	 int strides_5[1] = {1};
	load_mem(din_addr[3] + 0, 0x10000, 0, 0, 0, strides_5, sizes_5, 1, 0);
	config(0x0, 0, 0, 0);
	execute(0x4011c, 0, 0);
	 int sizes_10[1] = {512};
	 int strides_10[1] = {1};
	store_mem(dout_addr[0] + 0, 0x18000, 0, 0, strides_10, sizes_10, 1, 0);
fence(1);}
void cgra_execute(void** din_addr, void** dout_addr)
{
	 #ifdef CGRA_DEBUG
	 printf("|");
	 #endif
	static unsigned short cin[45][3] __attribute__((aligned(16))) = {
		{0x6000, 0x0002, 0x0010},
		{0x0080, 0x0000, 0x0011},
		{0x0000, 0x0000, 0x0012},
		{0x0000, 0x0000, 0x0013},
		{0x0000, 0x0000, 0x0014},
		{0x0000, 0x0000, 0x0015},
		{0x0000, 0x0200, 0x0016},
		{0x0000, 0x0000, 0x0017},
		{0x8000, 0x0002, 0x0028},
		{0x0080, 0x0000, 0x0029},
		{0x0000, 0x0000, 0x002a},
		{0x0000, 0x0000, 0x002b},
		{0x0000, 0x0000, 0x002c},
		{0x0000, 0x0000, 0x002d},
		{0x0000, 0x2e00, 0x002e},
		{0x0041, 0x0000, 0x002f},
		{0x4000, 0x0002, 0x0038},
		{0x0080, 0x0000, 0x0039},
		{0x0000, 0x0000, 0x003a},
		{0x0000, 0x0000, 0x003b},
		{0x0000, 0x0000, 0x003c},
		{0x0000, 0x0000, 0x003d},
		{0x0000, 0x0200, 0x003e},
		{0x0000, 0x0000, 0x003f},
		{0x0000, 0x0002, 0x0040},
		{0x0080, 0x0000, 0x0041},
		{0x0000, 0x0000, 0x0042},
		{0x0000, 0x0000, 0x0043},
		{0x0000, 0x0000, 0x0044},
		{0x0000, 0x0000, 0x0045},
		{0x0000, 0x0200, 0x0046},
		{0x0000, 0x0000, 0x0047},
		{0x2000, 0x0002, 0x0048},
		{0x0080, 0x0000, 0x0049},
		{0x0000, 0x0000, 0x004a},
		{0x0000, 0x0000, 0x004b},
		{0x0000, 0x0000, 0x004c},
		{0x0000, 0x0000, 0x004d},
		{0x0000, 0x0200, 0x004e},
		{0x0000, 0x0000, 0x004f},
		{0x2028, 0x0000, 0x0121},
		{0x0001, 0x0000, 0x0148},
		{0x000a, 0x0001, 0x0149},
		{0x1008, 0x0000, 0x0151},
		{0x1008, 0x0000, 0x0169},
	};

	load_cfg((void*)cin, 0x100000, 270, 0, 0);
	 int sizes_2[1] = {512};
	 int strides_2[1] = {1};
	load_mem(din_addr[0] + 0, 0x0, 0, 0, 0, strides_2, sizes_2, 1, 0);
	 int sizes_3[1] = {512};
	 int strides_3[1] = {1};
	load_mem(din_addr[1] + 0, 0x8000, 0, 0, 0, strides_3, sizes_3, 1, 0);
	 int sizes_4[1] = {512};
	 int strides_4[1] = {1};
	load_mem(din_addr[2] + 0, 0x10000, 0, 0, 0, strides_4, sizes_4, 1, 0);
	 int sizes_5[1] = {512};
	 int strides_5[1] = {1};
	load_mem(din_addr[3] + 0, 0x18000, 0, 0, 0, strides_5, sizes_5, 1, 0);
	config(0x0, 45, 0, 0);
	execute(0x1d2, 0, 0);
	 int sizes_10[1] = {512};
	 int strides_10[1] = {1};
	store_mem(dout_addr[0] + 0, 0x20000, 0, 0, strides_10, sizes_10, 1, 0);
fence(1);}
