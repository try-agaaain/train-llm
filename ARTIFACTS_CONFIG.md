# Artifacts 目录配置说明

## 概述

本项目支持灵活配置 artifacts 目录路径，优先级为：**命令行参数 > 环境变量 > 默认值**

## 配置方式

### 1. 命令行参数（最高优先级）

在 Makefile 命令中直接指定：

```bash
# 训练模型
make train ARTIFACTS_DIR=/custom/path

# 评估模型
make evaluate EVAL_BATCH_SIZE=8 ARTIFACTS_DIR=/custom/path

# 推送/拉取模型
make mpush ARTIFACTS_DIR=/custom/path
make mpull ARTIFACTS_DIR=/custom/path

# 推送/拉取数据集
make dpush ARTIFACTS_DIR=/custom/path
make dpull ARTIFACTS_DIR=/custom/path

# 启动服务器
make server ARTIFACTS_DIR=/custom/path
```

### 2. 环境变量（中等优先级）

在 `.env` 文件中配置：

```bash
# .env 文件
ARTIFACTS_DIR=/path/to/your/artifacts
```

或在运行命令前临时设置：

```bash
export ARTIFACTS_DIR=/custom/path
make train
```

### 3. 默认值（最低优先级）

如果未指定参数和环境变量，将使用默认路径：`./artifacts`（项目根目录下的 artifacts 文件夹）

## Kaggle 环境配置

在 Kaggle notebook 中，已经自动配置为：

- **代码和环境**：存放在 `/tmp/train-llm`（临时目录，不会保存）
- **Artifacts**：存放在 `/kaggle/working/artifacts`（持久化目录，会保存）

这样可以确保：
1. 代码和依赖安装不占用持久化空间
2. 只有重要的模型和数据集结果会被保存
3. 节省 Kaggle 的存储空间

## 目录结构

```
artifacts/
├── dataset/
│   ├── qa_dataset.json              # 训练数据集
│   ├── qa_dataset_padding.json      # 评估结果
│   ├── qa_dataset_augmented.json    # 增强数据集
│   └── book_split.jsonl             # 书籍切分结果
└── models/
    ├── local_finetuned/             # 训练阶段中间输出（checkpoint）
    └── qwen3_finetuned/             # 训练完成后的最终模型（用于上传/推理/测试）
```

## Python 脚本使用

所有 Python 脚本已更新为自动使用配置的 artifacts 目录：

```python
from utils import get_artifacts_dir

# 获取 artifacts 目录（自动按优先级查找）
artifacts_dir = get_artifacts_dir(current_file=__file__)

# 使用路径
model_path = artifacts_dir / "models" / "qwen3_finetuned"
dataset_path = artifacts_dir / "dataset" / "qa_dataset.json"
```

支持自定义路径（例如在命令行工具中）：

```python
# 优先使用命令行参数
artifacts_dir = get_artifacts_dir(
    custom_path=args.artifacts_dir,  # 来自 argparse
    current_file=__file__
)
```

## 受影响的文件

### 核心工具
- `utils/env_loader.py`：添加 `get_artifacts_dir()` 函数

### 配置文件
- `.env.example`：添加 `ARTIFACTS_DIR` 配置示例
- `makefile`：所有命令支持 `ARTIFACTS_DIR` 参数

### Python 脚本
- `train/train.py`：训练脚本
- `data/evaluate_model.py`：评估脚本（支持 `--artifacts-dir` 参数）
- `data/extract_qa.py`：QA 提取脚本
- `data/book_split.py`：书籍切分脚本
- `data/augment_dataset.py`：数据增强脚本
- `data/generate_qa.py`：QA 生成脚本
- `data/merge_dataset.py`：数据集合并脚本

### Notebook
- `kaggle/train-llm.ipynb`：Kaggle 训练脚本（优化目录结构）

## 示例

### 本地开发（使用默认路径）

```bash
# 使用默认 ./artifacts
make train
make evaluate
```

### 使用环境变量

```bash
# 设置环境变量
echo "ARTIFACTS_DIR=/data/my-artifacts" >> .env

# 运行命令（自动使用环境变量）
make train
make evaluate
```

### 使用命令行参数

```bash
# 临时指定路径（不修改配置文件）
make train ARTIFACTS_DIR=/tmp/test-artifacts
make evaluate ARTIFACTS_DIR=/tmp/test-artifacts
```

### Kaggle 环境

Notebook 会自动设置环境变量：

```python
os.environ['ARTIFACTS_DIR'] = '/kaggle/working/artifacts'
```

所有后续命令都会使用这个路径。

## 迁移指南

如果你之前的脚本使用了硬编码路径，现在可以：

1. **不做任何修改**：使用默认的 `./artifacts` 路径
2. **设置环境变量**：在 `.env` 中配置 `ARTIFACTS_DIR`
3. **使用参数**：在 make 命令中传递 `ARTIFACTS_DIR=/custom/path`

所有方式都完全兼容！
