# NexOS-AI (`lab2`)

NexOS 是一套用于操作系统课程实验的 RISC-V 教学内核，运行在 QEMU `virt` 机器上。
当前分支是 `lab2`，它提供了 AI 相关路径：

## 1. 你会用到什么

- Linux 环境，推荐 Ubuntu / Debian
- `qemu-system-riscv64`
- 一套 RISC-V 交叉工具链，满足其一即可
- `riscv64-unknown-elf-*`
- `riscv64-linux-gnu-*`
- `python3`
- `unzip`

如果你想在宿主机上做文本和 token id 的互转，还需要：

- Python 包 `tokenizers`

`Makefile` 会自动探测工具链和 QEMU。如果本机没有这些依赖，构建会直接报错。

## 2. 模型准备

这条分支默认只使用提供的预导出模型权重文件，不要求你自己从 HuggingFace 原始权重重新导出。

下载地址：

- SmolLM2-135M: `https://git.ustc.edu.cn/KONC/os-lab/-/raw/main/smol.zip?ref_type=heads&inline=false`

为了实验体验，本次实验的 `qemu-llm` 主线只使用 `Smol`。仓库里虽然保留了 `QWEN` 的 tokenizer 文件，方便做宿主机上的文本 / token 转换或后续扩展，但主线启动不需要 `qwen.zip`。

准备方式：

1. 在仓库根目录创建 `models/`
2. 把压缩包放进去，并保持文件名为 `smol.zip`
3. 不需要手动解压，直接使用 `make qemu-llm` 即可；`Makefile` 会在需要时自动解压模型资产、生成镜像并启动 QEMU

推荐目录结构：

```text
Spring2026OS/
├── Makefile
├── README.md
├── models/
│   └── smol.zip
└── ...
```

解压后，宿主机会得到：

```text
models/
├── smol.zip
├── SMOL/
│   ├── INFO.TXT
│   ├── CFG.BIN
│   ├── EMB.BIN
│   ├── NRM.BIN
│   ├── ROP.BIN
│   ├── PMT.BIN
│   ├── EXP.BIN
│   ├── L00.BIN
│   └── ...
```

说明：

- `smol.zip` 需要能解出 `SMOL/INFO.TXT`
- 当前 `qemu-llm` 主线会把 `SMOL/` 打包进镜像里的 `/AI/SMOL`

如果想覆盖默认模型目录，也可以在 `make` 时传参：

```bash
make qemu-llm PREMODEL_ROOT=/path/to/models
```

## 3. 如何启动


```bash
make qemu-llm
```

它会：

1. 编译内核和用户程序
2. 在需要时自动解压 `models/smol.zip`
3. 生成只包含 `/AI/SMOL` 的镜像
4. 按当前 `Makefile` 的 QEMU 配置启动系统


注意：

- 如果你还没有补完 `ai_service_worker_register()` / `ai_service_worker_get()` / `ai_service_worker_complete()`，启动日志里仍可能看到 `ai_daemon` 注册失败


## 4. 进入 guest 后怎么跑

成功启动后你会进入系统 shell，提示符通常是 `$ `。可以先执行：


```shell
$ aitest
[aitest] tokens: 0
[aitest] token_count: 1
[aitest] predict_count: 1
[aitest] result: sync-smoke predict=1 prompt=0
[aitest] bytes: 29
```

退出 QEMU 的方式：

```text
Ctrl+A, 然后按 X
```

### 4.2 关于 AI service 路径

`lab2` 比 `AI` 多了一条内核 service 线：

- `kernel/core/ai_service.c`
- `user/ai_daemon.c`
- `user/aitest.c`
- `user/syscall.c` 里的 `ai_call` / `ai_submit` / `ai_wait` / `ai_query`

## 5. 宿主机文本/token 转换工具

为了配合 `aitest` 或读 guest 输出，仓库里已经带了宿主机 tokenizer 工具：

- `tools/translate_llm_io.py`
- `tools/tokenizers/SMOL/tokenizer.json`

当前实验主线请优先使用 `--model smol`。

依赖：

```bash
python3 -m pip install --user tokenizers
```

如果你的系统还没有 `pip`，例如 Ubuntu / Debian，可以先安装：

```bash
sudo apt install python3-pip
python3 -m pip install --user tokenizers
```

文本编码为 token id：

```bash
python3 tools/translate_llm_io.py encode --model smol --text "The capital of France is"
```

token id 反解码为文本：

```bash
python3 tools/translate_llm_io.py decode --model smol --tokens "504 3575 282 4649 314"
```

如果你后面自己扩展到 `QWEN`，只需要把上面的 `--model smol` 换成 `--model qwen` 即可。

如果你想覆盖默认 tokenizer 文件，也可以显式指定：

```bash
python3 tools/translate_llm_io.py encode --model smol --tokenizer-file /path/to/tokenizer.json --text "hello"
```

这套工具最适合做两件事：

- 先把自然语言 prompt 编码成 token id，再喂给 guest 里的 `aitest`
- 把 guest 输出的 token id 反解码回文本
