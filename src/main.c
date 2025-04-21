#include "DRAMAddr.hpp"
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

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <emmintrin.h>
#include <immintrin.h>
#include <random>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <x86intrin.h>
#include <sched.h>

const size_t N_PAGES = 400;
const size_t N_SAMPLES_PER_AVG = 128;
const size_t N_AVGS_PER_MEASUREMENT = 17;
const size_t THRESHOLD_SAMPLES = 1024 * 5;
const size_t N_BUCKETS = 200;
const size_t THRESHOLD_WINDOW_SIZE = 5;
const size_t PAGE_2MB = 1024 * 1024 * 2;
static size_t PAGE_SIZE = PAGE_2MB;

typedef struct {
  double_t lower_bound;
  double_t upper_bound;
} area;

typedef struct {
  size_t start;
  size_t end;
  bool high_activity;
} activity;

typedef struct {
  area ar;
  uint32_t count;
} bucket;

void* allocate(size_t n_pages) {
  void* res = mmap(NULL, n_pages * PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  if(res == MAP_FAILED) {
    printf("memory allocation failed (%s)\n", strerror(errno));
    exit(EXIT_FAILURE);
  }
  memset(res, 0x42, n_pages * PAGE_SIZE);
  return res;
}

uint64_t median(std::vector<uint64_t> &vec) {
  size_t n = vec.size() / 2;
  std::nth_element(vec.begin(), vec.begin() + n, vec.end());
  return vec[n];
}

uint32_t sum(bucket buckets[], size_t n_buckets) {
  uint32_t sum = 0;
  for(size_t i = 0; i < n_buckets; i++) {
    sum += buckets[i].count;
  }
  return sum;
}

double_t avg(bucket buckets[], size_t n_buckets) {
  return sum(buckets, n_buckets) / (double_t)n_buckets;
}

void* random_page_addr(void *start_addr, size_t alloc_size, size_t step_size) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> distr(0, (alloc_size * PAGE_SIZE / step_size) - 1);

  size_t random = distr(gen) * step_size;

  return start_addr + random;
}

uint64_t measure_timing(volatile char *addr1, volatile char *addr2) {
  unsigned int tsc_aux;
  std::vector<uint64_t> avgs;
  avgs.reserve(N_AVGS_PER_MEASUREMENT);

  for(int i = 0; i < N_AVGS_PER_MEASUREMENT; i++) {
    sched_yield();
    uint64_t sum = 0;

    for(int j = 0; j < N_SAMPLES_PER_AVG; j++) {
      _mm_clflushopt((void *)addr1);
      _mm_clflushopt((void *)addr2);
      _mm_mfence();

      uint64_t start = __rdtscp(&tsc_aux);
      _mm_mfence();
      *addr1;
      *addr2;
      _mm_mfence();
      uint64_t end = __rdtscp(&tsc_aux);

      sum += end - start;
    }
    
    avgs.push_back(sum / N_SAMPLES_PER_AVG);
  }
  
  return median(avgs);
}

bucket* to_hist(std::vector<uint64_t> access_times, size_t n_buckets) {
  bucket* hist = new bucket[n_buckets];
  std::sort(access_times.begin(), access_times.end());
  uint64_t min = access_times.front();
  uint64_t max = access_times.back();
  double_t window_size = ((double_t)(max - min)) / n_buckets;
  size_t current_bucket = 0;
  double_t current_window = min + window_size;
  hist[0].ar = {current_window - window_size, current_window};

  for(size_t i = 0; i < access_times.size(); i++) {
    while(access_times[i] > current_window && current_bucket + 1 < n_buckets) {
      current_bucket++;
      current_window += window_size;
      hist[current_bucket].ar = {current_window - window_size, current_window};
    }
    hist[current_bucket].count = hist[current_bucket].count + 1;
  }

  return hist;
}

std::vector<activity> find_major_areas(bucket histogram[], size_t buckets) {
  double_t threshold = avg(histogram, buckets) / 2;

  size_t start = 0;
  uint32_t window_sum = sum(histogram, THRESHOLD_WINDOW_SIZE);
  bool above = ((double_t)window_sum / THRESHOLD_WINDOW_SIZE) > threshold;

  std::vector<activity> res;

  for(size_t i = 1; i < buckets - THRESHOLD_WINDOW_SIZE; i++) {
    window_sum -= histogram[i - 1].count;
    window_sum += histogram[i + THRESHOLD_WINDOW_SIZE - 1].count;
    bool current_above = (double_t)window_sum / THRESHOLD_WINDOW_SIZE > threshold;
    if(current_above != above && i != start) {
      res.push_back({start, i, above});
      above = current_above;
      start = i + 1;
    }
  }

  res.push_back({start, buckets - 1, above});

  for(activity a : res) {
    printf("area from %lu to %lu with activity %s\n", a.start, a.end, a.high_activity ? "HIGH" : "LOW");
  }

  return res;
}

uint64_t find_conflict_threshold(void *alloc_start, size_t allocated_pages, size_t step_size) {
  std::vector<uint64_t> samples;
  samples.reserve(THRESHOLD_SAMPLES);

  for(int i = 0; i < THRESHOLD_SAMPLES; i++) {
    auto addr1 = (volatile char*)random_page_addr(alloc_start, allocated_pages, step_size);
    auto addr2 = (volatile char*)random_page_addr(alloc_start, allocated_pages, step_size);
    auto access_time = measure_timing(addr1, addr2);
    samples.push_back(access_time);
  }

  printf("created %lu samples", samples.size());

  auto hist = to_hist(samples, N_BUCKETS);
  std::sort(samples.begin(), samples.end());
  printf("theoretical threshold for 16 banks is %lu\n", samples[samples.size() * (15.0 / 16.0)]);
  for(size_t i = 0; i < N_BUCKETS; i++) {
    printf("hist %lu at %f to %f: %d\n", i, hist[i].ar.lower_bound, hist[i].ar.upper_bound, hist[i].count);
  }

  std::vector<activity> areas = find_major_areas(hist, N_BUCKETS);
  if(areas.size() < 2) {
    printf("error: only able to identify %lu areas, expected at least 2.\n", areas.size());
    exit(EXIT_FAILURE);
  }

  size_t i = 0;
  while(i < areas.size() && !areas[i].high_activity) {
    i++;
  }
  while(i < areas.size() && areas[i].high_activity) {
    i++;
  }

  if(i >= areas.size()) {
    printf("unable to identify area of high activity. Something has gone horribly wrong.\n");
    exit(EXIT_FAILURE);
  }

  activity threshold_act = areas[i];
  double_t area_start = hist[threshold_act.start].ar.upper_bound;
  double_t area_end = hist[threshold_act.end].ar.lower_bound;

  uint64_t threshold = area_start + (area_end - area_start) / 2;

  uint32_t above_threshold = 0;
  for(auto sample : samples) {
    if(sample > threshold) {
      above_threshold++;
    }
  }

  printf("ratio was %f\n", ((double_t)above_threshold) / samples.size());

  return threshold;
}


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
  DRAMAddr::initialize_mapping(0, mem.buffer[0]);

	uint64_t thresh = find_conflict_threshold(mem.buffer[0], N_PAGES, PAGE_SIZE / 2);
  
  DRAMAddr a1 = DRAMAddr(mem.buffer[0]);
  DRAMAddr a2 = a1.add(0, 1, 0);

  uint64_t timing = measure_timing((volatile char *)a1.to_virt(), (volatile char *)a2.to_virt());
  assert(timing > thresh);

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
