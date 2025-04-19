#include "DRAMConfig.hpp"
#include "stdio.h"
// #include <x86intrin.h> /* for rdtsc, rdtscp, clflush */
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdbool.h>
#include <getopt.h>
#include <stdint.h>

#include "include/utils.h"
#include "include/types.h"
#include "include/allocator.h"
#include "include/memory.h"
#include "include/hammer-suite.h"
#include "include/params.h"

ProfileParams *p;

int main(int argc, char **argv)
{
  DRAMConfig::select_config(Microarchitecture::AMD_ZEN_3, 1, 4, 4, false);
	srand(time(NULL));
	p = (ProfileParams*)malloc(sizeof(ProfileParams));
	if (p == NULL) {
		fprintf(stderr, "[ERROR] Memory allocation\n");
		exit(1);
	}

	if(process_argv(argc, argv, p) == -1) {
		free(p);
		exit(1);
	}
	assert(USE_1GB ^ USE_THP);

	MemoryBuffer mem = {
		.buffer = (char **)malloc(sizeof(char *) * NUM_PAGES),
		.physmap = NULL,
		.fd = p->huge_fd,

#if USE_1GB
		.size = p->m_size,
#endif
#if USE_THP
		.size = HUGE_SIZE * NUM_PAGES,
#endif
		.align = p->m_align,
		.flags = p->g_flags & MEM_MASK
	};

	alloc_buffer(&mem);
	set_physmap(&mem);

	SessionConfig s_cfg;
	memset(&s_cfg, 0, sizeof(SessionConfig));

	s_cfg.h_rows = PATT_LEN;
	s_cfg.h_rounds = p->rounds;
	s_cfg.h_cfg = N_SIDED;
	s_cfg.d_cfg = FILL_FF ? ONE_TO_ZERO : ZERO_TO_ONE;
	s_cfg.base_off = p->base_off;
	s_cfg.aggr_n = p->aggr;
	s_cfg.dist = p->dist;
	s_cfg.vics = p->vics;

#if USE_THP
	mem_check(&s_cfg, &mem);
#endif
#if USE_1GB
	mem_check_1GB(&s_cfg, &mem);
#endif

	return 0;
}
