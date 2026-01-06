# 环境设置
VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
MODELSCOPE_USER=SoFarSoLong
MODEL_NAME=qwen3_merged
DATASET_NAME=qa_dataset

# Artifacts目录配置 (优先级: 命令行参数 > 环境变量 > 默认值)
ARTIFACTS_DIR ?= $(shell echo $${ARTIFACTS_DIR:-./artifacts})

# 默认目标
.DEFAULT_GOAL := help

# 帮助信息
help:
	@echo "可用命令:"
	@echo "  make setup        初始化开发环境"
	@echo "  make train        训练模型"
	@echo "  make test         运行测试"
	@echo "  make clean        清理生成文件"
	@echo ""
	@echo "  模型管理:"
	@echo "  make mpush        推送模型到ModelScope"
	@echo "  make mpull        从ModelScope拉取模型"
	@echo ""
	@echo "  数据集管理:"
	@echo "  make dpush        推送数据集到ModelScope"
	@echo "  make dpull        从ModelScope拉取数据集"
	@echo ""
	@echo "  make evaluate     评估模型"
	@echo "  make server       启动VLLM服务器"
	@echo ""
	@echo "  Artifacts目录配置 (优先级: 参数 > 环境变量 > 默认值):"
	@echo "    ARTIFACTS_DIR   指定artifacts目录 (默认: ./artifacts)"
	@echo "  示例: make train ARTIFACTS_DIR=/custom/path"
	@echo "  详见: ARTIFACTS_CONFIG.md"

# 初始化环境
UV_INDEX_URL ?=
setup:
	$(PIP) install uv
	if [ -n "$(UV_INDEX_URL)" ]; then \
	  uv sync --index-url $(UV_INDEX_URL); \
	else \
	  uv sync; \
	fi

# 训练模型
train:
	ARTIFACTS_DIR=$(ARTIFACTS_DIR) $(PYTHON) train/train.py

# 运行测试
test:
	$(PYTHON) -m pytest train/test.py

# 清理
clean:
	rm -rf __pycache__ .pytest_cache
	find . -name '*.pyc' -delete

# 评估模型
# 指定 batch size：make evaluate EVAL_BATCH_SIZE=4
# 指定 artifacts 目录：make evaluate ARTIFACTS_DIR=/custom/path
# 传递更多参数：make evaluate ARGS="--max-tokens 512 --output result.json"
EVAL_BATCH_SIZE ?= 8
evaluate:
	$(PYTHON) data/evaluate_model.py --batch-size $(EVAL_BATCH_SIZE) --artifacts-dir $(ARTIFACTS_DIR) $(ARGS)

# 推送模型到ModelScope
PUSH_MODEL_DIR := $(ARTIFACTS_DIR)/models/qwen3_merged_for_upload

mpush: setup-model-upload-dir
	@echo "正在推送模型到ModelScope..."
	uv run modelscope upload \
		$(MODELSCOPE_USER)/$(MODEL_NAME) \
		$(PUSH_MODEL_DIR) \
		--repo-type model \
		--commit-message "模型更新" \
		--token $(MODELSCOPE_TOKEN) \
		|| (echo "推送失败，请检查：1. modelscope是否安装 2. 环境变量是否正确设置"; exit 1)
	@echo "清理临时目录..."
	rm -rf $(PUSH_MODEL_DIR)
	@echo "推送成功！访问地址：https://modelscope.cn/$(MODELSCOPE_USER)/$(MODEL_NAME)"

# 准备模型上传目录
setup-model-upload-dir:
	@echo "正在准备上传目录 $(PUSH_MODEL_DIR)..."
	rm -rf $(PUSH_MODEL_DIR)
	mkdir -p $(PUSH_MODEL_DIR)
	# 复制必要的 LoRA/Adapter 文件
	cp $(ARTIFACTS_DIR)/models/qwen3_merged/{adapter_config.json,adapter_model.safetensors,merges.txt,added_tokens.json,chat_template.jinja,special_tokens_map.json,tokenizer.json,tokenizer_config.json,vocab.json,README.md} $(PUSH_MODEL_DIR)/
	@echo "上传目录准备完毕。"

mpull:
	@echo "正在清理本地旧模型目录..."
	rm -rf $(ARTIFACTS_DIR)/models/qwen3_finetuned
	rm -rf $(ARTIFACTS_DIR)/models/qwen3_merged
	@echo "正在从 ModelScope 下载模型..."
	mkdir -p $(ARTIFACTS_DIR)/models
	# 使用 modelscope download 命令的 standard 格式
	uv run modelscope download \
		--model $(MODELSCOPE_USER)/$(MODEL_NAME) \
		--local_dir $(ARTIFACTS_DIR)/models/qwen3_merged \
		--token $(MODELSCOPE_TOKEN) \
		|| (echo "拉取失败，请检查：1. 网络连接 2. 模型 ID 是否正确 3. Token 是否有效"; exit 1)
	@echo "下载完成！模型已保存在 $(ARTIFACTS_DIR)/models/qwen3_merged"

server:
	python -m vllm.entrypoints.openai.api_server \
	--model $(ARTIFACTS_DIR)/models/qwen3_merged \
	--served-model-name qwen-ft \
	--trust-remote-code \
	--gpu-memory-utilization 0.9 \
	--port 8000

# 推送数据集到ModelScope
PUSH_DATASET_DIR := $(ARTIFACTS_DIR)/dataset_for_upload

dpush: setup-dataset-upload-dir
	@echo "正在推送数据集到ModelScope..."
	uv run modelscope upload \
		$(MODELSCOPE_USER)/$(DATASET_NAME) \
		$(PUSH_DATASET_DIR) \
		--repo-type dataset \
		--commit-message "数据集更新" \
		--token $(MODELSCOPE_TOKEN) \
		|| (echo "推送失败，请检查：1. modelscope是否安装 2. 环境变量是否正确设置"; exit 1)
	@echo "清理临时目录..."
	rm -rf $(PUSH_DATASET_DIR)
	@echo "推送成功！访问地址：https://modelscope.cn/datasets/$(MODELSCOPE_USER)/$(DATASET_NAME)"

# 准备数据集上传目录
setup-dataset-upload-dir:
	@echo "正在准备数据集上传目录 $(PUSH_DATASET_DIR)..."
	rm -rf $(PUSH_DATASET_DIR)
	mkdir -p $(PUSH_DATASET_DIR)
	# 复制数据集文件
	cp $(ARTIFACTS_DIR)/dataset/*.json $(PUSH_DATASET_DIR)/ 2>/dev/null || true
	cp $(ARTIFACTS_DIR)/dataset/*.jsonl $(PUSH_DATASET_DIR)/ 2>/dev/null || true
	# 创建README
	@echo "# QA Dataset\n\n本数据集包含计算机网络相关的问答对，用于微调语言模型。\n\n## 文件说明\n\n- qa_dataset.json: 主要训练数据集\n- qa_dataset_padding.json: 模型评估结果\n- qa_dataset_augmented.json: 增强后的数据集\n" > $(PUSH_DATASET_DIR)/README.md
	@echo "数据集上传目录准备完毕。"

dpull:
	@echo "正在从 ModelScope 下载数据集..."
	mkdir -p $(ARTIFACTS_DIR)/dataset
	# 使用 modelscope download 命令下载数据集
	uv run modelscope download \
		--model $(MODELSCOPE_USER)/$(DATASET_NAME) \
		--local_dir $(ARTIFACTS_DIR)/dataset \
		--token $(MODELSCOPE_TOKEN) \
		|| (echo "拉取失败，请检查：1. 网络连接 2. 数据集 ID 是否正确 3. Token 是否有效"; exit 1)
	@echo "下载完成！数据集已保存在 $(ARTIFACTS_DIR)/dataset"

kinit:
	@echo "正在初始化 Kaggle Notebook 元数据..."
	uv run kaggle kernels init -p kaggle
	@echo "元数据文件已在 kaggle/kernel-metadata.json 创建。请编辑此文件以更新 Notebook 信息。"

kpush:
	@echo "正在推送 Jupyter Notebook 到 Kaggle..."
	uv run kaggle kernels push -p kaggle
	@echo "推送完成！"

kpull:
	@echo "正在从 Kaggle 拉取 Jupyter Notebook..."
	uv run kaggle kernels pull team317/train-llm -p ./kaggle -m
	@echo "拉取完成！"

kstatus:
	@echo "正在检查 Kaggle Notebook 状态..."
	uv run kaggle kernels status team317/train-llm
	@echo "状态检查完成！"

koutput:
	@echo "正在获取 Kaggle Notebook 输出..."
	uv run kaggle kernels output team317/train-llm -p ./kaggle/output
	@echo "输出已保存到 ./kaggle_output 目录！"

.PHONY: help setup train test clean mpush mpull dpush dpull setup-model-upload-dir setup-dataset-upload-dir server kinit kpush kpull kstatus koutput