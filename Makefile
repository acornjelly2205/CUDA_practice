

# 사용법
#   make                      # ROOT 아래 모든 하위 폴더 빌드
#   make v0                   # 특정 폴더만 빌드 (폴더 이름으로)
#   make ARCH=sm_75           # T4: sm_75 / 4090: sm_89 / 5090: sm_120
#   make run ARGS=4096        # 빌드 후 전부 실행
#   make list                 # 빌드 대상 목록 확인
#   make clean

ROOT      = .
COMMON    ?= common
BUILD     ?= build
ARCH      ?= sm_89

NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -arch=$(ARCH)

# depth=1 하위 폴더 중 main.cu가 있는 것만 대상
SRCS  := $(wildcard $(ROOT)/*/main.cu)
NAMES := $(patsubst $(ROOT)/%/main.cu,%,$(SRCS))
BINS  := $(addprefix $(BUILD)/,$(NAMES))

COMMON_HDRS := $(wildcard $(COMMON)/*.cuh $(COMMON)/*.h)

.PHONY: all run list clean $(NAMES)

all: $(BINS)

# 폴더 이름으로 개별 빌드: make v0
$(NAMES): %: $(BUILD)/%

# 각 폴더의 main.cu + 같은 폴더의 *.cuh + common 헤더가 바뀌면 재빌드
.SECONDEXPANSION:
$(BUILD)/%: $(ROOT)/%/main.cu $$(wildcard $(ROOT)/$$*/*.cuh) $(COMMON_HDRS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -I$(ROOT)/$* -I$(COMMON) -o $@ $<

$(BUILD):
	mkdir -p $@

run: all
	@for b in $(BINS); do echo "===== $$b"; ./$$b $(ARGS) || exit 1; done

list:
	@echo "ROOT  = $(ROOT)"
	@echo "NAMES = $(NAMES)"

clean:
	rm -rf $(BUILD)