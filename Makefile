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
RAM ?= 1536M

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

QEMUGDB = -S -gdb tcp::26000

# C flags
CFLAGS = -Wall -Werror -O -fno-omit-frame-pointer -ggdb
CFLAGS += -mcmodel=medany -mno-relax
CFLAGS += -ffreestanding -fno-common -nostdlib
CFLAGS += -fno-pie -no-pie
CFLAGS += -MMD -MP
CFLAGS += -I$(KINCLUDE)

# Debug logger level: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR, 4=NONE
LOG_LEVEL ?= 3
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
COMDBDIR = $(BUILD)/compdb
COMDB = $(BUILD)/compile_commands.json

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

UPROGS = init sh hello quiet stressio stsched stressdisk pid uptime sleep killer kill pingpong fstat forktest zombie echo cat wc grep ls find xargs fstest mkdir rm ln touch logtest llmrun_smol llmrun_qwen ai_daemon aitest
UCOMMON = $(UBUILD)/entry.o $(UBUILD)/syscall.o $(UBUILD)/printf.o $(UBUILD)/ulib.o
UOBJS = $(UCOMMON) $(LLMRUN_SHARED) $(patsubst %,$(UBUILD)/%.o,$(UPROGS))
UELFS = $(patsubst %,$(UBUILD)/%.elf,$(UPROGS))
LLMRUN_SHARED = $(UBUILD)/llmrun_support.o

PREMODEL_ROOT ?= models
SMOL_ARCHIVE ?= $(PREMODEL_ROOT)/smol.zip
SMOL_ASSET_DIR ?= $(PREMODEL_ROOT)/SMOL
SMOL_ASSET_STAMP ?= $(SMOL_ASSET_DIR)/INFO.TXT
LLM_FSIMG ?= build/fs-llm.img
LLM_FSIMG_BLOCKS ?= 21875
LLM_FSIMG_NINODES ?= 4096

define extract_model_zip
	@set -e; \
	archive="$(1)"; \
	model_name="$(2)"; \
	dest_dir="$(3)"; \
	dest_stamp="$(4)"; \
	if [ ! -f "$$archive" ]; then \
		echo "Missing teacher-provided $$model_name archive."; \
		echo "Expected file: $$archive"; \
		echo "Please download $$model_name as a zip file into $(PREMODEL_ROOT) and rerun make."; \
		false; \
	fi; \
	mkdir -p "$(PREMODEL_ROOT)"; \
	tmpdir=$$(mktemp -d "$(PREMODEL_ROOT)/.$$model_name.extract.XXXXXX"); \
	trap 'rm -rf "$$tmpdir"' EXIT INT TERM; \
	echo "[model-assets] extracting $$archive -> $$dest_dir"; \
	unzip -q -o "$$archive" -d "$$tmpdir"; \
	src_stamp=$$(find "$$tmpdir" -type f -path "*/$$model_name/INFO.TXT" | head -n 1); \
	if [ -z "$$src_stamp" ]; then \
		echo "Archive $$archive does not contain $$model_name/INFO.TXT."; \
		false; \
	fi; \
	src_dir=$$(dirname "$$src_stamp"); \
	rm -rf "$$dest_dir"; \
	mv "$$src_dir" "$$dest_dir"; \
	test -f "$$dest_stamp"; \
	trap - EXIT INT TERM; \
	rm -rf "$$tmpdir"; \
	echo "[model-assets] ready at $$dest_dir"
endef

# Default target
all: kernel.elf $(COMDB)

# Build rules
$(BUILD)/%.o: %.c
	@mkdir -p $(@D) $(COMDBDIR)
	COMPDB_DIR=$(COMDBDIR) COMPDB_FILE=$< $(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/%.o: %.S
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

$(UBUILD)/llmrun_smol.o: $(U)/llmrun_smol.c $(U)/llmrun_support.h
	@mkdir -p $(@D)
	$(REALCC) $(LLMRUN_UCFLAGS) -c -o $@ $<

$(UBUILD)/llmrun_qwen.o: $(U)/llmrun_qwen.c $(U)/llmrun_support.h
	@mkdir -p $(@D)
	$(REALCC) $(LLMRUN_UCFLAGS) -c -o $@ $<

$(UBUILD)/ai_daemon.o: $(U)/ai_daemon.c $(U)/llmrun_support.h
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

$(UBUILD)/ai_daemon.elf: $(UCOMMON) $(UBUILD)/ai_daemon.o $(LLMRUN_SHARED) $(U)/user.ld
	$(REALCC) $(LLMRUN_UCFLAGS) $(ULDFLAGS) -T $(U)/user.ld -o $@ $(UCOMMON) $(UBUILD)/ai_daemon.o $(LLMRUN_SHARED)

# Run in QEMU
qemu: kernel.elf $(COMDB) fsimg
	$(QEMU) $(QEMUOPTS) $(QEMUFSOPTS)

fs: kernel.elf $(COMDB) fsimg
	$(QEMU) $(QEMUOPTS) $(QEMUFSOPTS)

fsimg: $(UELFS) tools/mkfsimg.py
	python3 tools/mkfsimg.py --image fs.img --size-blocks 2048 --add init=$(UBUILD)/init.elf --add sh=$(UBUILD)/sh.elf --add hello=$(UBUILD)/hello.elf --add quiet=$(UBUILD)/quiet.elf --add stressio=$(UBUILD)/stressio.elf --add stsched=$(UBUILD)/stsched.elf --add stressdisk=$(UBUILD)/stressdisk.elf --add pid=$(UBUILD)/pid.elf --add uptime=$(UBUILD)/uptime.elf --add sleep=$(UBUILD)/sleep.elf --add killer=$(UBUILD)/killer.elf --add kill=$(UBUILD)/kill.elf --add pingpong=$(UBUILD)/pingpong.elf --add fstat=$(UBUILD)/fstat.elf --add forktest=$(UBUILD)/forktest.elf --add zombie=$(UBUILD)/zombie.elf --add echo=$(UBUILD)/echo.elf --add cat=$(UBUILD)/cat.elf --add wc=$(UBUILD)/wc.elf --add grep=$(UBUILD)/grep.elf --add ls=$(UBUILD)/ls.elf --add find=$(UBUILD)/find.elf --add xargs=$(UBUILD)/xargs.elf --add fstest=$(UBUILD)/fstest.elf --add mkdir=$(UBUILD)/mkdir.elf --add rm=$(UBUILD)/rm.elf --add ln=$(UBUILD)/ln.elf --add touch=$(UBUILD)/touch.elf --add logtest=$(UBUILD)/logtest.elf --add llmrun=$(UBUILD)/llmrun_smol.elf --add llmrun_smol=$(UBUILD)/llmrun_smol.elf --add llmrun_qwen=$(UBUILD)/llmrun_qwen.elf --add ai_daemon=$(UBUILD)/ai_daemon.elf --add aitest=$(UBUILD)/aitest.elf

llm-assets: $(SMOL_ASSET_STAMP)
	@echo "[llm-assets] using Smol assets from $(SMOL_ASSET_DIR)"

$(SMOL_ASSET_STAMP):
	$(call extract_model_zip,$(SMOL_ARCHIVE),SMOL,$(SMOL_ASSET_DIR),$(SMOL_ASSET_STAMP))

fsimg-llm: $(LLM_FSIMG)

$(LLM_FSIMG): $(UELFS) tools/mkfsimg.py Makefile $(SMOL_ASSET_STAMP)
	python3 tools/mkfsimg.py --image $(LLM_FSIMG) --size-blocks $(LLM_FSIMG_BLOCKS) --ninodes $(LLM_FSIMG_NINODES) --add init=$(UBUILD)/init.elf --add sh=$(UBUILD)/sh.elf --add hello=$(UBUILD)/hello.elf --add quiet=$(UBUILD)/quiet.elf --add stressio=$(UBUILD)/stressio.elf --add stsched=$(UBUILD)/stsched.elf --add stressdisk=$(UBUILD)/stressdisk.elf --add pid=$(UBUILD)/pid.elf --add uptime=$(UBUILD)/uptime.elf --add sleep=$(UBUILD)/sleep.elf --add killer=$(UBUILD)/killer.elf --add kill=$(UBUILD)/kill.elf --add pingpong=$(UBUILD)/pingpong.elf --add fstat=$(UBUILD)/fstat.elf --add forktest=$(UBUILD)/forktest.elf --add zombie=$(UBUILD)/zombie.elf --add echo=$(UBUILD)/echo.elf --add cat=$(UBUILD)/cat.elf --add wc=$(UBUILD)/wc.elf --add grep=$(UBUILD)/grep.elf --add ls=$(UBUILD)/ls.elf --add find=$(UBUILD)/find.elf --add xargs=$(UBUILD)/xargs.elf --add fstest=$(UBUILD)/fstest.elf --add mkdir=$(UBUILD)/mkdir.elf --add rm=$(UBUILD)/rm.elf --add ln=$(UBUILD)/ln.elf --add touch=$(UBUILD)/touch.elf --add logtest=$(UBUILD)/logtest.elf --add llmrun=$(UBUILD)/llmrun_smol.elf --add llmrun_smol=$(UBUILD)/llmrun_smol.elf --add ai_daemon=$(UBUILD)/ai_daemon.elf --add aitest=$(UBUILD)/aitest.elf --add-dir AI/SMOL=$(SMOL_ASSET_DIR)

qemu-gdb: kernel.elf $(COMDB) fsimg
	$(QEMU) $(QEMUOPTS) $(QEMUFSOPTS) $(QEMUGDB)

qemu-llm: kernel.elf $(COMDB) fsimg-llm
	$(QEMU) $(QEMUOPTS) -drive file=$(LLM_FSIMG),if=none,format=raw,id=x0 -device virtio-blk-device,drive=x0,bus=virtio-mmio-bus.0

gdb: kernel.elf fsimg
	$(QEMU) $(QEMUOPTS) $(QEMUFSOPTS) $(QEMUGDB) & \
	$(GDB) kernel.elf -ex "target remote localhost:26000"

kernel.asm: kernel.elf
	$(OBJDUMP) -d kernel.elf > kernel.asm

$(COMDB): $(KOBJS)
	@mkdir -p $(BUILD)
	@python3 tools/merge_compdb.py $(COMDBDIR) $(COMDB)

compdb: $(COMDB)

clean:
	rm -rf $(BUILD) kernel.elf kernel.asm fs.img

.PHONY: all qemu fs fsimg llm-assets fsimg-llm qemu-llm qemu-gdb gdb clean compdb

-include $(KOBJS:.o=.d) $(UOBJS:.o=.d)
