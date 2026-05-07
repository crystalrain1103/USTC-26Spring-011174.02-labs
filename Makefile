# Simple kernel build

K = kernel
BUILD = build

KINCLUDE = $(K)/include
KARCH = $(K)/arch/riscv
KCORE = $(K)/core
KDRV = $(K)/drivers
KLIB = $(K)/lib
KLD = $(K)/ld
KFS = $(K)/fs

CPUS ?= 4
PHYSTOP_MB ?= 128
RAM ?= $(PHYSTOP_MB)M
LAZY_ALLOC ?= 1
COW_ALLOC ?= 1
QEMU_TEST_TIMEOUT ?= 40
AI_PHYSTOP_MB ?= 1536
FSIMG_ALL ?= build/fs-all.img

U = user
UBUILD = $(BUILD)/user

# Toolchain auto-detect
ifndef TOOLPREFIX
TOOLPREFIX := $(shell if riscv64-unknown-elf-gcc -print-libgcc-file-name > /dev/null 2>&1; \
	then echo "riscv64-unknown-elf-"; \
	elif riscv64-linux-gnu-gcc -print-libgcc-file-name > /dev/null 2>&1; \
	then echo "riscv64-linux-gnu-"; \
	else echo "***"; fi)
endif

ifeq ($(TOOLPREFIX),***)
$(error "Couldn't find a RISC-V toolchain in PATH")
endif

REALCC = $(TOOLPREFIX)gcc
CC = python3 tools/ccwrap.py $(REALCC)
LD = $(TOOLPREFIX)ld
OBJCOPY = $(TOOLPREFIX)objcopy
OBJDUMP = $(TOOLPREFIX)objdump
GDB = $(TOOLPREFIX)gdb

# QEMU auto-detect
ifndef QEMU
QEMU := $(shell if which qemu-system-riscv64 > /dev/null; \
	then echo qemu-system-riscv64; \
	else echo "***"; fi)
endif

ifeq ($(QEMU),***)
$(error "qemu-system-riscv64 not found in PATH")
endif

QEMUOPTS = -machine virt \
	-bios none \
	-kernel kernel.elf \
	-m $(RAM) \
	-smp $(CPUS) \
	-nographic \
	-global virtio-mmio.force-legacy=false
QEMUFSOPTS = -drive file=fs.img,if=none,format=raw,id=x0 -device virtio-blk-device,drive=x0,bus=virtio-mmio-bus.0
QEMUFSOPTS_ALL = -drive file=$(FSIMG_ALL),if=none,format=raw,id=x0 -device virtio-blk-device,drive=x0,bus=virtio-mmio-bus.0

QEMUGDB = -S -gdb tcp::26000

# C flags
CFLAGS = -Wall -Werror -O -fno-omit-frame-pointer -ggdb
CFLAGS += -mcmodel=medany -mno-relax
CFLAGS += -ffreestanding -fno-common -nostdlib
CFLAGS += -fno-pie -no-pie
CFLAGS += -MMD -MP
CFLAGS += -I$(KINCLUDE)
CFLAGS += -DLAZY_ALLOC=$(LAZY_ALLOC) -DCOW_ALLOC=$(COW_ALLOC)
CFLAGS += -DPHYSTOP_MB=$(PHYSTOP_MB)

# Debug logger level: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR, 4=NONE
LOG_LEVEL ?= 2
CFLAGS += -DLOG_LEVEL=$(LOG_LEVEL)

UCFLAGS = -Wall -Werror -O -fno-omit-frame-pointer -ggdb
UCFLAGS += -mcmodel=medany -mno-relax
UCFLAGS += -ffreestanding -fno-common -nostdlib
UCFLAGS += -fno-pie -no-pie
UCFLAGS += -MMD -MP
UCFLAGS += -I$(KINCLUDE) -I$(U)
ULDFLAGS = -Wl,--build-id=none
LLMRUN_UCFLAGS = $(UCFLAGS) -O3 -funroll-loops

LDFLAGS = -z max-page-size=4096 --no-warn-rwx-segments
COMDBDIR = .compdb
COMDB = compile_commands.json
FEATURE_STAMP = $(BUILD)/.feature-flags

# Kernel sources
SRCS = \
	$(KARCH)/entry.S \
	$(KARCH)/kernelvec.S \
	$(KARCH)/trampoline.S \
	$(KARCH)/timervec.S \
	$(KARCH)/swtch.S \
	$(KARCH)/start.c \
	$(KDRV)/plic.c \
	$(KDRV)/uart.c \
	$(KDRV)/virtio_disk.c \
	$(KLIB)/string.c \
	$(KLIB)/printf.c \
	$(KCORE)/ai_service.c \
	$(KCORE)/trap.c \
	$(KCORE)/cpu.c \
	$(KCORE)/intr.c \
	$(KCORE)/spinlock.c \
	$(KCORE)/sleeplock.c \
	$(KCORE)/kalloc.c \
	$(KCORE)/vm.c \
	$(KCORE)/console.c \
	$(KCORE)/file.c \
	$(KCORE)/proc.c \
	$(KCORE)/main.c \
	$(KCORE)/syscall.c \
	$(KCORE)/pipe.c \
	$(KCORE)/bio.c \
	$(KFS)/fs.c \

KOBJS = $(patsubst %.c,$(BUILD)/%.o,$(filter %.c,$(SRCS)))
KOBJS += $(patsubst %.S,$(BUILD)/%.o,$(filter %.S,$(SRCS)))

BASE_UPROGS = \
	init \
	sh \
	hello \
	quiet \
	stressio \
	stsched \
	stressdisk \
	pid \
	uptime \
	sleep \
	killer \
	kill \
	pingpong \
	fstat \
	forktest \
	zombie \
	echo \
	cat \
	wc \
	grep \
	ls \
	find \
	xargs \
	fstest \
	mkdir \
	rm \
	ln \
	touch \
	logtest

AI_UPROGS = \
	llmrun_smol \
	llmrun_qwen \
	aitest \
	smolprobe

LAB3_UPROGS = \
	lazytest \
	mmaptest \
	crash \
	cowtest

UPROGS = $(BASE_UPROGS) $(AI_UPROGS) $(LAB3_UPROGS)
UCOMMON = $(UBUILD)/entry.o $(UBUILD)/syscall.o $(UBUILD)/printf.o $(UBUILD)/ulib.o
UOBJS = $(UCOMMON) $(patsubst %,$(UBUILD)/%.o,$(UPROGS))
UELFS = $(patsubst %,$(UBUILD)/%.elf,$(UPROGS))
LAB3_UELFS = $(patsubst %,$(UBUILD)/%.elf,$(BASE_UPROGS) $(LAB3_UPROGS))
LLMRUN_SHARED = $(UBUILD)/llmrun_support.o

POWERSERVE_PY ?= /home/llm/.conda/envs/cyliu/bin/python
# By default, use the pre-exported NexOS SmolLM assets in models/.
# Set LLM_MODEL_DIR=/path/to/raw/hf/model and LLM_ASSET_DIR=build/llm/smolfs
# if you want Make to regenerate the BIN assets from config.json/model.safetensors.
LLM_MODEL_DIR ?=
LLM_ASSET_DIR ?= models/SmolLM2-135M-Instruct
LLM_ASSET_STAMP ?= $(LLM_ASSET_DIR)/INFO.TXT
LLM_FSIMG ?= build/fs-llm.img
LLM_FSIMG_EXTRA_ARGS ?=
LLM_PROMPT ?= The capital of France is
LLM_PREDICT ?= 1
LLM_MULTI_PROMPT ?= Answer in one short phrase. The capital of France is
LLM_MULTI_PREDICT ?= 4
LLM_MULTI_ASSET_DIR ?= $(BUILD)/llm/smolfs-multi
LLM_MULTI_ASSET_STAMP ?= $(LLM_MULTI_ASSET_DIR)/INFO.TXT
LLM_MULTI_FSIMG ?= build/fs-llm-multi.img
AITEST_REQ_JSON ?= docs/aitest_req.json
LLM_AITEST_FSIMG ?= build/fs-llm-aitest.img
QWEN_MODEL_DIR ?= models/Qwen3.5-0.8B
QWEN_ASSET_DIR ?= $(BUILD)/llm/qwenfs
QWEN_ASSET_STAMP ?= $(QWEN_ASSET_DIR)/INFO.TXT
QWEN_PROMPT ?= The capital of France is
QWEN_PREDICT ?= 1
QWEN_RUNTIME_SEQ ?= 32
QWEN_FSIMG ?= build/fs-qwen.img
QWEN_BONUS_ASSET_DIR ?= $(BUILD)/llm/qwen-bonus
QWEN_BONUS_ASSET_STAMP ?= $(QWEN_BONUS_ASSET_DIR)/INFO.TXT
QWEN_BONUS_REPORT ?= build/qwen3_bonus_report.json

MMAP_BONUS_TEST_CMDS = mmaptest\n

# Default target
all: kernel.elf $(COMDB)

# Build rules
FORCE:

$(FEATURE_STAMP): FORCE
	@mkdir -p $(BUILD)
	@tmp=$@.tmp; \
	printf "LAZY_ALLOC=%s\nCOW_ALLOC=%s\nPHYSTOP_MB=%s\n" "$(LAZY_ALLOC)" "$(COW_ALLOC)" "$(PHYSTOP_MB)" > $$tmp; \
	if ! test -f $@ || ! cmp -s $$tmp $@; then mv $$tmp $@; else rm -f $$tmp; fi

$(BUILD)/%.o: %.c $(FEATURE_STAMP)
	@mkdir -p $(@D) $(COMDBDIR)
	COMPDB_DIR=$(COMDBDIR) COMPDB_FILE=$< $(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/%.o: %.S $(FEATURE_STAMP)
	@mkdir -p $(@D) $(COMDBDIR)
	COMPDB_DIR=$(COMDBDIR) COMPDB_FILE=$< $(CC) $(CFLAGS) -c -o $@ $<

# Link kernel
kernel.elf: $(KLD)/kernel.ld $(KOBJS) | $(COMDB)
	$(LD) $(LDFLAGS) -T $(KLD)/kernel.ld -o $@ $(KOBJS)

# ------------------------------------------------------------
# User programs
# ------------------------------------------------------------

$(UBUILD)/%.o: $(U)/%.c
	@mkdir -p $(@D)
	$(REALCC) $(UCFLAGS) -c -o $@ $<

$(UBUILD)/llmrun_smol.o: $(U)/llmrun_smol.c
	@mkdir -p $(@D)
	$(REALCC) $(LLMRUN_UCFLAGS) -c -o $@ $<

$(UBUILD)/llmrun_qwen.o: $(U)/llmrun_qwen.c
	@mkdir -p $(@D)
	$(REALCC) $(LLMRUN_UCFLAGS) -c -o $@ $<

$(UBUILD)/smolprobe.o: $(U)/smolprobe.c $(U)/llmrun_smol.c
	@mkdir -p $(@D)
	$(REALCC) $(LLMRUN_UCFLAGS) -c -o $@ $<

$(UBUILD)/llmrun_support.o: $(U)/llmrun_support.c $(U)/llmrun_support.h
	@mkdir -p $(@D)
	$(REALCC) $(LLMRUN_UCFLAGS) -c -o $@ $<

$(UBUILD)/%.o: $(U)/%.S
	@mkdir -p $(@D)
	$(REALCC) $(UCFLAGS) -c -o $@ $<

$(UBUILD)/%.elf: $(UCOMMON) $(UBUILD)/%.o $(U)/user.ld
	$(REALCC) $(UCFLAGS) $(ULDFLAGS) -T $(U)/user.ld -o $@ $(UCOMMON) $(UBUILD)/$*.o

$(UBUILD)/llmrun_smol.elf: $(UCOMMON) $(UBUILD)/llmrun_smol.o $(LLMRUN_SHARED) $(U)/user.ld
	$(REALCC) $(LLMRUN_UCFLAGS) $(ULDFLAGS) -T $(U)/user.ld -o $@ $(UCOMMON) $(UBUILD)/llmrun_smol.o $(LLMRUN_SHARED)

$(UBUILD)/llmrun_qwen.elf: $(UCOMMON) $(UBUILD)/llmrun_qwen.o $(LLMRUN_SHARED) $(U)/user.ld
	$(REALCC) $(LLMRUN_UCFLAGS) $(ULDFLAGS) -T $(U)/user.ld -o $@ $(UCOMMON) $(UBUILD)/llmrun_qwen.o $(LLMRUN_SHARED)

$(UBUILD)/smolprobe.elf: $(UCOMMON) $(UBUILD)/smolprobe.o $(LLMRUN_SHARED) $(U)/user.ld
	$(REALCC) $(LLMRUN_UCFLAGS) $(ULDFLAGS) -T $(U)/user.ld -o $@ $(UCOMMON) $(UBUILD)/smolprobe.o $(LLMRUN_SHARED)

# Run in QEMU
qemu: kernel.elf $(COMDB) fsimg
	$(QEMU) $(QEMUOPTS) $(QEMUFSOPTS)

qemu-all: kernel.elf $(COMDB) fsimg-all
	$(QEMU) $(QEMUOPTS) $(QEMUFSOPTS_ALL)

fs: kernel.elf $(COMDB) fsimg
	$(QEMU) $(QEMUOPTS) $(QEMUFSOPTS)

fsimg: $(LAB3_UELFS) tools/mkfsimg.py
	python3 tools/mkfsimg.py \
		--image fs.img \
		--size-blocks 65536 \
		--add init=$(UBUILD)/init.elf \
		--add sh=$(UBUILD)/sh.elf \
		--add hello=$(UBUILD)/hello.elf \
		--add quiet=$(UBUILD)/quiet.elf \
		--add stressio=$(UBUILD)/stressio.elf \
		--add stsched=$(UBUILD)/stsched.elf \
		--add stressdisk=$(UBUILD)/stressdisk.elf \
		--add pid=$(UBUILD)/pid.elf \
		--add uptime=$(UBUILD)/uptime.elf \
		--add sleep=$(UBUILD)/sleep.elf \
		--add killer=$(UBUILD)/killer.elf \
		--add kill=$(UBUILD)/kill.elf \
		--add pingpong=$(UBUILD)/pingpong.elf \
		--add fstat=$(UBUILD)/fstat.elf \
		--add forktest=$(UBUILD)/forktest.elf \
		--add zombie=$(UBUILD)/zombie.elf \
		--add echo=$(UBUILD)/echo.elf \
		--add cat=$(UBUILD)/cat.elf \
		--add wc=$(UBUILD)/wc.elf \
		--add grep=$(UBUILD)/grep.elf \
		--add ls=$(UBUILD)/ls.elf \
		--add find=$(UBUILD)/find.elf \
		--add xargs=$(UBUILD)/xargs.elf \
		--add fstest=$(UBUILD)/fstest.elf \
		--add mkdir=$(UBUILD)/mkdir.elf \
		--add rm=$(UBUILD)/rm.elf \
		--add ln=$(UBUILD)/ln.elf \
		--add touch=$(UBUILD)/touch.elf \
		--add logtest=$(UBUILD)/logtest.elf \
		--add lazytest=$(UBUILD)/lazytest.elf \
		--add mmaptest=$(UBUILD)/mmaptest.elf \
		--add crash=$(UBUILD)/crash.elf \
		--add cowtest=$(UBUILD)/cowtest.elf

fsimg-all: $(UELFS) tools/mkfsimg.py
	python3 tools/mkfsimg.py \
		--image $(FSIMG_ALL) \
		--size-blocks 65536 \
		--add init=$(UBUILD)/init.elf \
		--add sh=$(UBUILD)/sh.elf \
		--add hello=$(UBUILD)/hello.elf \
		--add quiet=$(UBUILD)/quiet.elf \
		--add stressio=$(UBUILD)/stressio.elf \
		--add stsched=$(UBUILD)/stsched.elf \
		--add stressdisk=$(UBUILD)/stressdisk.elf \
		--add pid=$(UBUILD)/pid.elf \
		--add uptime=$(UBUILD)/uptime.elf \
		--add sleep=$(UBUILD)/sleep.elf \
		--add killer=$(UBUILD)/killer.elf \
		--add kill=$(UBUILD)/kill.elf \
		--add pingpong=$(UBUILD)/pingpong.elf \
		--add fstat=$(UBUILD)/fstat.elf \
		--add forktest=$(UBUILD)/forktest.elf \
		--add zombie=$(UBUILD)/zombie.elf \
		--add echo=$(UBUILD)/echo.elf \
		--add cat=$(UBUILD)/cat.elf \
		--add wc=$(UBUILD)/wc.elf \
		--add grep=$(UBUILD)/grep.elf \
		--add ls=$(UBUILD)/ls.elf \
		--add find=$(UBUILD)/find.elf \
		--add xargs=$(UBUILD)/xargs.elf \
		--add fstest=$(UBUILD)/fstest.elf \
		--add mkdir=$(UBUILD)/mkdir.elf \
		--add rm=$(UBUILD)/rm.elf \
		--add ln=$(UBUILD)/ln.elf \
		--add touch=$(UBUILD)/touch.elf \
		--add logtest=$(UBUILD)/logtest.elf \
		--add llmrun=$(UBUILD)/llmrun_smol.elf \
		--add llmrun_smol=$(UBUILD)/llmrun_smol.elf \
		--add llmrun_qwen=$(UBUILD)/llmrun_qwen.elf \
		--add aitest=$(UBUILD)/aitest.elf \
		--add smolprobe=$(UBUILD)/smolprobe.elf \
		--add smol_probe=$(UBUILD)/smolprobe.elf \
		--add lazytest=$(UBUILD)/lazytest.elf \
		--add mmaptest=$(UBUILD)/mmaptest.elf \
		--add crash=$(UBUILD)/crash.elf \
		--add cowtest=$(UBUILD)/cowtest.elf

qemu-all fsimg-all qemu-llm qemu-llm-multi qemu-llm-aitest fsimg-llm fsimg-llm-multi fsimg-llm-aitest fsimg-qwen qemu-qwen-native: PHYSTOP_MB=$(AI_PHYSTOP_MB)
qemu-all fsimg-all qemu-llm qemu-llm-multi qemu-llm-aitest fsimg-llm fsimg-llm-multi fsimg-llm-aitest fsimg-qwen qemu-qwen-native: RAM=$(AI_PHYSTOP_MB)M

ifeq ($(strip $(LLM_MODEL_DIR)),)
llm-assets: $(LLM_ASSET_STAMP)

$(LLM_ASSET_STAMP):
	@printf "missing pre-exported SmolLM assets at %s\n" "$(LLM_ASSET_DIR)"
	@printf "set LLM_ASSET_DIR=/path/to/exported/assets, or set LLM_MODEL_DIR=/path/to/raw/model to export assets\n"
	@exit 1
else
llm-assets: $(LLM_ASSET_STAMP)

$(LLM_ASSET_STAMP): tools/export_smollm.py $(LLM_MODEL_DIR)/config.json $(LLM_MODEL_DIR)/model.safetensors
	@mkdir -p $(dir $(LLM_ASSET_DIR))
	$(POWERSERVE_PY) tools/export_smollm.py --model-dir $(LLM_MODEL_DIR) --out-dir $(LLM_ASSET_DIR) --prompt "$(LLM_PROMPT)" --predict $(LLM_PREDICT)
endif

llm-assets-multi:
	$(MAKE) llm-assets LLM_ASSET_DIR=$(LLM_MULTI_ASSET_DIR) LLM_ASSET_STAMP=$(LLM_MULTI_ASSET_STAMP) LLM_PROMPT='$(LLM_MULTI_PROMPT)' LLM_PREDICT=$(LLM_MULTI_PREDICT)

fsimg-llm: $(LLM_FSIMG)

$(LLM_FSIMG): $(UELFS) tools/mkfsimg.py $(LLM_ASSET_STAMP)
	python3 tools/mkfsimg.py --image $(LLM_FSIMG) --size-blocks 21875 --ninodes 4096 --add init=$(UBUILD)/init.elf --add sh=$(UBUILD)/sh.elf --add hello=$(UBUILD)/hello.elf --add quiet=$(UBUILD)/quiet.elf --add stressio=$(UBUILD)/stressio.elf --add stsched=$(UBUILD)/stsched.elf --add stressdisk=$(UBUILD)/stressdisk.elf --add pid=$(UBUILD)/pid.elf --add uptime=$(UBUILD)/uptime.elf --add sleep=$(UBUILD)/sleep.elf --add killer=$(UBUILD)/killer.elf --add kill=$(UBUILD)/kill.elf --add pingpong=$(UBUILD)/pingpong.elf --add fstat=$(UBUILD)/fstat.elf --add forktest=$(UBUILD)/forktest.elf --add zombie=$(UBUILD)/zombie.elf --add echo=$(UBUILD)/echo.elf --add cat=$(UBUILD)/cat.elf --add wc=$(UBUILD)/wc.elf --add grep=$(UBUILD)/grep.elf --add ls=$(UBUILD)/ls.elf --add find=$(UBUILD)/find.elf --add xargs=$(UBUILD)/xargs.elf --add fstest=$(UBUILD)/fstest.elf --add mkdir=$(UBUILD)/mkdir.elf --add rm=$(UBUILD)/rm.elf --add ln=$(UBUILD)/ln.elf --add touch=$(UBUILD)/touch.elf --add logtest=$(UBUILD)/logtest.elf --add llmrun=$(UBUILD)/llmrun_smol.elf --add llmrun_smol=$(UBUILD)/llmrun_smol.elf --add llmrun_qwen=$(UBUILD)/llmrun_qwen.elf --add aitest=$(UBUILD)/aitest.elf --add smolprobe=$(UBUILD)/smolprobe.elf --add smol_probe=$(UBUILD)/smolprobe.elf --add-dir $(LLM_ASSET_DIR) $(LLM_FSIMG_EXTRA_ARGS)

fsimg-llm-multi:
	$(MAKE) fsimg-llm LLM_ASSET_DIR=$(LLM_MULTI_ASSET_DIR) LLM_ASSET_STAMP=$(LLM_MULTI_ASSET_STAMP) LLM_FSIMG=$(LLM_MULTI_FSIMG) LLM_PROMPT='$(LLM_MULTI_PROMPT)' LLM_PREDICT=$(LLM_MULTI_PREDICT)

fsimg-llm-aitest:
	$(MAKE) fsimg-llm LLM_FSIMG=$(LLM_AITEST_FSIMG) LLM_FSIMG_EXTRA_ARGS="--add req.json=$(AITEST_REQ_JSON)"

qemu-gdb: kernel.elf $(COMDB) fsimg
	$(QEMU) $(QEMUOPTS) $(QEMUFSOPTS) $(QEMUGDB)

qemu-llm: kernel.elf $(COMDB) fsimg-llm
	$(QEMU) $(QEMUOPTS) -drive file=$(LLM_FSIMG),if=none,format=raw,id=x0 -device virtio-blk-device,drive=x0,bus=virtio-mmio-bus.0

qemu-llm-multi: kernel.elf $(COMDB) fsimg-llm-multi
	$(QEMU) $(QEMUOPTS) -drive file=$(LLM_MULTI_FSIMG),if=none,format=raw,id=x0 -device virtio-blk-device,drive=x0,bus=virtio-mmio-bus.0

qemu-llm-aitest: kernel.elf $(COMDB) fsimg-llm-aitest
	$(QEMU) $(QEMUOPTS) -drive file=$(LLM_AITEST_FSIMG),if=none,format=raw,id=x0 -device virtio-blk-device,drive=x0,bus=virtio-mmio-bus.0

qwen-assets: $(QWEN_ASSET_STAMP)

$(QWEN_ASSET_STAMP): tools/export_qwen.py $(QWEN_MODEL_DIR)/config.json $(QWEN_MODEL_DIR)/model.safetensors.index.json
	@mkdir -p $(dir $(QWEN_ASSET_DIR))
	$(POWERSERVE_PY) tools/export_qwen.py --model-dir $(QWEN_MODEL_DIR) --out-dir $(QWEN_ASSET_DIR) --prompt "$(QWEN_PROMPT)" --predict $(QWEN_PREDICT) --runtime-seq-len $(QWEN_RUNTIME_SEQ)

fsimg-qwen: $(QWEN_FSIMG)

$(QWEN_FSIMG): $(UELFS) tools/mkfsimg.py $(QWEN_ASSET_STAMP)
	python3 tools/mkfsimg.py --image $(QWEN_FSIMG) --size-blocks 62500 --ninodes 4096 --add init=$(UBUILD)/init.elf --add sh=$(UBUILD)/sh.elf --add hello=$(UBUILD)/hello.elf --add quiet=$(UBUILD)/quiet.elf --add stressio=$(UBUILD)/stressio.elf --add stsched=$(UBUILD)/stsched.elf --add stressdisk=$(UBUILD)/stressdisk.elf --add pid=$(UBUILD)/pid.elf --add uptime=$(UBUILD)/uptime.elf --add sleep=$(UBUILD)/sleep.elf --add killer=$(UBUILD)/killer.elf --add kill=$(UBUILD)/kill.elf --add pingpong=$(UBUILD)/pingpong.elf --add fstat=$(UBUILD)/fstat.elf --add forktest=$(UBUILD)/forktest.elf --add zombie=$(UBUILD)/zombie.elf --add echo=$(UBUILD)/echo.elf --add cat=$(UBUILD)/cat.elf --add wc=$(UBUILD)/wc.elf --add grep=$(UBUILD)/grep.elf --add ls=$(UBUILD)/ls.elf --add find=$(UBUILD)/find.elf --add xargs=$(UBUILD)/xargs.elf --add fstest=$(UBUILD)/fstest.elf --add mkdir=$(UBUILD)/mkdir.elf --add rm=$(UBUILD)/rm.elf --add ln=$(UBUILD)/ln.elf --add touch=$(UBUILD)/touch.elf --add logtest=$(UBUILD)/logtest.elf --add llmrun=$(UBUILD)/llmrun_smol.elf --add llmrun_smol=$(UBUILD)/llmrun_smol.elf --add llmrun_qwen=$(UBUILD)/llmrun_qwen.elf --add aitest=$(UBUILD)/aitest.elf --add-dir $(QWEN_ASSET_DIR)

qemu-qwen-native: kernel.elf $(COMDB) fsimg-qwen
	$(QEMU) $(QEMUOPTS) -drive file=$(QWEN_FSIMG),if=none,format=raw,id=x0 -device virtio-blk-device,drive=x0,bus=virtio-mmio-bus.0

qwen-bonus-probe: tools/qwen_bonus_probe.py
	@mkdir -p $(dir $(QWEN_BONUS_REPORT))
	python3 tools/qwen_bonus_probe.py --model-dir $(QWEN_MODEL_DIR) --out $(QWEN_BONUS_REPORT)

qwen-bonus-assets: $(QWEN_BONUS_ASSET_STAMP)

$(QWEN_BONUS_ASSET_STAMP): tools/export_qwen_bonus.py $(QWEN_MODEL_DIR)/config.json $(QWEN_MODEL_DIR)/model.safetensors.index.json
	@mkdir -p $(dir $(QWEN_BONUS_ASSET_DIR))
	python3 tools/export_qwen_bonus.py --model-dir $(QWEN_MODEL_DIR) --out-dir $(QWEN_BONUS_ASSET_DIR)

gdb: kernel.elf fsimg
	$(QEMU) $(QEMUOPTS) $(QEMUFSOPTS) $(QEMUGDB) & \
	$(GDB) kernel.elf -ex "target remote localhost:26000"

kernel.asm: kernel.elf
	$(OBJDUMP) -d kernel.elf > kernel.asm

$(COMDB): $(KOBJS)
	@mkdir -p $(COMDBDIR)
	@python3 tools/merge_compdb.py $(COMDBDIR) $(COMDB)

compdb: $(COMDB)

print-config:
	@printf "LAZY_ALLOC=%s\nCOW_ALLOC=%s\nCPUS=%s\nRAM=%s\n" \
		"$(LAZY_ALLOC)" "$(COW_ALLOC)" "$(CPUS)" "$(RAM)"

mmap-bonus-test: kernel.elf $(COMDB) fsimg
	@mkdir -p $(BUILD)
	@test "$(LAZY_ALLOC)" = "1" || { echo "mmap-bonus-test requires LAZY_ALLOC=1"; exit 1; }
	@{ printf '%b' "$(MMAP_BONUS_TEST_CMDS)"; sleep 1; } | \
		timeout $(QEMU_TEST_TIMEOUT)s $(QEMU) $(QEMUOPTS) $(QEMUFSOPTS) \
		> $(BUILD)/mmap-bonus-test.log 2>&1 || test $$? -eq 124
	@grep -q "All mmap tests passed" $(BUILD)/mmap-bonus-test.log
	@! grep -Eq "\\[FAIL\\]|mappages: remap|panic:|kerneltrap" $(BUILD)/mmap-bonus-test.log
	@printf "mmap bonus tests passed; log: %s\n" "$(BUILD)/mmap-bonus-test.log"

clean:
	rm -rf $(BUILD) $(COMDBDIR) kernel.elf kernel.asm fs.img

.PHONY: FORCE all qemu qemu-all fs fsimg fsimg-all llm-assets llm-assets-multi fsimg-llm fsimg-llm-multi fsimg-llm-aitest fsimg-qwen qwen-assets qemu-llm qemu-llm-multi qemu-llm-aitest qemu-qwen-native qwen-bonus-probe qwen-bonus-assets qemu-gdb gdb clean compdb print-config mmap-bonus-test

-include $(KOBJS:.o=.d) $(UOBJS:.o=.d)
