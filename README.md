# 训练代码

本目录只包含预训练、环境配置、启动脚本和少量冒烟测试图像。建议在 Linux + NVIDIA GPU 环境运行。

## 1. 获取代码

```bash
git clone https://github.com/jiZhangji/Ph.git
cd Ph
```

## 2. 硬件与软件

推荐配置：

- 2×NVIDIA A100 80GB 或更高
- Linux x86_64
- NVIDIA Driver 支持 CUDA 12.1
- Conda 或 Mamba
- 充足的本地 SSD 空间；单个完整 checkpoint 可能超过 1GB

创建环境：

```bash
conda env create -f environment.yml
conda activate sar-pretrain
```

如果服务器只能使用已有 PyTorch 环境，请先安装与服务器 CUDA 匹配的 PyTorch/torchvision，再执行：

```bash
pip install -r requirements.txt
```

检查 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name()); print(torch.cuda.is_bf16_supported())"
```

## 3. 数据目录

数据加载器递归读取 PNG、JPG、JPEG、TIF、TIFF 和 BMP，并统一转换为单通道。类别目录不是必需的。

```text
dataset/
├── source_a/
│   ├── image_0001.png
│   └── image_0002.png
└── source_b/
    └── image_0003.tif
```

仓库附带的 `dataset/SARSim` 仅用于冒烟测试，不能替代正式预训练数据。

## 4. 先运行冒烟测试

单 GPU：

```bash
bash scripts/smoke_pretrain.sh
```

CPU 仅用于验证代码路径：

```bash
bash scripts/smoke_pretrain_cpu.sh
```

## 5. 探测 2×A100 最大 batch size

最大 batch size 取决于 A100 显存容量、驱动、PyTorch、是否编译扩展以及同卡是否存在其他进程，不能写死。正式全量训练前建议先用合成输入和完整前向/反向/优化器步骤进行二分探测：

```bash
python scripts/find_max_batch_size.py \
  --gpus 2 \
  --model mae_vit_base_patch16 \
  --upper 2048
```

输出示例：

```text
MAX_BATCH_SIZE_PER_GPU=160
RECOMMENDED_BATCH_SIZE_PER_GPU=144
GLOBAL_BATCH_AT_MAX=320
```

`MAX` 是探测时的容量上限；长时间训练建议使用 `RECOMMENDED`，为数据加载波动和显存碎片预留约 10%。如果必须使用容量上限，可直接采用 `MAX`。

如果目标机器比较空闲，也可以把 `--upper` 继续提高到 3072：

```bash
python scripts/find_max_batch_size.py --gpus 2 --upper 3072
```

## 6. 2×A100 80GB 全量正式预训练

脚本默认运行 300 epoch、2 进程 DDP、BF16、ViT-Base、每卡 batch size 为 512。A100 80GB 通常还有继续增大的空间，但正式全量训练前仍建议先用上一节探测出的 `RECOMMENDED_BATCH_SIZE_PER_GPU` 覆盖默认值。

```bash
export DATA_PATH=/path/to/full_dataset
export OUTPUT_DIR=/path/to/output_phyd_mae
export BATCH_SIZE=512
bash scripts/pretrain_2xa100.sh
```

脚本使用：

- 2 进程 DistributedDataParallel；
- BF16 自动混合精度；
- TF32 矩阵计算；
- 每 GPU 独立 batch size；
- 自动按全局 batch size 缩放学习率。

全局 batch size 为：

```text
BATCH_SIZE × 2 × accum_iter
```

如果显存仍然充足，可以提高每卡 batch：

```bash
export BATCH_SIZE=768
bash scripts/pretrain_2xa100.sh
```

如果显存不足，降低每卡 batch 或使用梯度累积：

```bash
export BATCH_SIZE=128
export ACCUM_ITER=2
bash scripts/pretrain_2xa100.sh
```

从 checkpoint 恢复：

```bash
torchrun --standalone --nproc_per_node=2 Pretraining/main_pretrain.py \
  --data_path /path/to/full_dataset \
  --output_dir /path/to/output \
  --log_dir /path/to/output \
  --batch_size 512 \
  --amp_dtype bf16 \
  --resume /path/to/checkpoint.pth
```

可通过环境变量覆盖脚本默认值：

```bash
export MODEL=mae_vit_base_patch16
export EPOCHS=300
export BATCH_SIZE=512
export ACCUM_ITER=1
export AMP_DTYPE=bf16
export BLR=1e-3
export WARMUP_EPOCHS=20
export SAVE_FREQ=50
bash scripts/pretrain_2xa100.sh
```

## 7. 4×A100 可选脚本

如果后续使用 4 卡服务器，仍可使用：

```bash
export DATA_PATH=/path/to/full_dataset
export OUTPUT_DIR=/path/to/output_phyd_mae_4gpu
export BATCH_SIZE=128
bash scripts/pretrain_4xa100.sh
```

## 8. 可选编译相对位置编码扩展

不编译也能运行；编译后通常更快。必须在目标服务器和最终 PyTorch 环境中执行：

```bash
cd Pretraining/rpe_ops
python setup.py build_ext --inplace
cd ../..
```

编译失败时可继续使用 Python 回退实现。

## 9. 常见问题

### CUDA out of memory

重新探测或降低每 GPU batch：

```bash
export BATCH_SIZE=64
bash scripts/pretrain_2xa100.sh
```

### 数据集小于全局 batch

训练使用 `drop_last=True`。正式数据的图像数量必须不小于全局 batch，否则一个 epoch 可能没有有效 batch。

### NCCL 初始化失败

确认两张 GPU 可见：

```bash
CUDA_VISIBLE_DEVICES=0,1 nvidia-smi
```

单机脚本使用 `torchrun --standalone`，不需要手工设置主节点地址。

### TensorBoard

```bash
tensorboard --logdir /path/to/output --port 6006
```

## 10. ModelScope dataset download

Download and extract the pretraining and classification datasets:

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/Ph

pip install -U modelscope
python scripts/download_modelscope_data.py
```

The default output paths are:

```text
dataset/modelscope/zips/Pretraining_dataset.zip
dataset/modelscope/zips/classification_dataset.zip
dataset/modelscope/extracted/Pretraining_dataset
dataset/modelscope/extracted/classification_dataset
```

Check that code and data are ready:

```bash
bash scripts/check_code_and_data.sh
```

## 11. 2xH100 pretraining

Run pretraining on two H100 GPUs:

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/Ph

export CUDA_VISIBLE_DEVICES=0,1
export DATA_PATH=dataset/modelscope/extracted/Pretraining_dataset
export OUTPUT_DIR=runs/pretrain_2xh100
export BATCH_SIZE=512

bash scripts/pretrain_2xh100.sh
```

If CUDA OOM appears, reduce the per-GPU batch size and rerun:

```bash
export BATCH_SIZE=256
bash scripts/pretrain_2xh100.sh
```

The H100 script uses `torchrun --standalone --nproc_per_node=2`, BF16 mixed
precision, TF32 matrix math enabled in `Pretraining/main_pretrain.py`, and DDP
with one process per GPU.
