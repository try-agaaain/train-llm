# 环境设置
VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
MODELSCOPE_USER=SoFarSoLong
MODEL_NAME=qwen3_merged
# 默认目标
.DEFAULT_GOAL := help

# 帮助信息
help:
	@echo "可用命令:"
	@echo "  make setup	   初始化开发环境"
	@echo "  make train	   训练模型"
	@echo "  make test		运行测试"
	@echo "  make clean	   清理生成文件"
	@echo "  make push-model  推送模型到ModelScope"

# 初始化环境
setup:
	$(PIP) install uv
	uv sync

# 训练模型
train:
	$(PYTHON) train/train.py

# 运行测试
test:
	$(PYTHON) -m pytest train/test.py

# 清理
clean:
	rm -rf __pycache__ .pytest_cache
	find . -name '*.pyc' -delete

# 评估模型
# 指定 batch size：make evaluate EVAL_BATCH_SIZE=4
# 传递更多参数：make evaluate ARGS="--max-tokens 512 --output result.json"
EVAL_BATCH_SIZE ?= 8
evaluate:
	$(PYTHON) data/evaluate_model.py --batch-size $(EVAL_BATCH_SIZE) $(ARGS)

# 推送模型到ModelScope
PUSH_DIR := ./qwen3_merged_for_upload

push-model: setup-upload-dir
	@echo "正在推送模型到ModelScope..."
	uv run modelscope upload \
		$(MODELSCOPE_USER)/$(MODEL_NAME) \
		$(PUSH_DIR) \
		--repo-type model \
		--commit-message "模型更新" \
		--token $(MODELSCOPE_TOKEN) \
		|| (echo "推送失败，请检查：1. modelscope是否安装 2. 环境变量是否正确设置"; exit 1)
	@echo "清理临时目录..."
	rm -rf $(PUSH_DIR)
	@echo "推送成功！访问地址：https://modelscope.cn/$(MODELSCOPE_USER)/$(MODEL_NAME)"

# 新增：准备上传目录
setup-upload-dir:
	@echo "正在准备上传目录 $(PUSH_DIR)..."
	rm -rf $(PUSH_DIR)
	mkdir -p $(PUSH_DIR)
	# 复制必要的 LoRA/Adapter 文件
	cp ./qwen3_merged/{adapter_config.json,adapter_model.safetensors,merges.txt,added_tokens.json,chat_template.jinja,special_tokens_map.json,tokenizer.json,tokenizer_config.json,vocab.json,README.md} $(PUSH_DIR)/
	@echo "上传目录准备完毕。"

pull:
	@echo "正在清理本地旧模型目录..."
	rm -rf ./qwen3_finetuned
	rm -rf ./qwen3_merged
	@echo "正在从 ModelScope 下载模型..."
	# 使用 modelscope download 命令的 standard 格式
	uv run modelscope download \
		--model $(MODELSCOPE_USER)/$(MODEL_NAME) \
		--local_dir ./qwen3_merged \
		--token $(MODELSCOPE_TOKEN) \
		|| (echo "拉取失败，请检查：1. 网络连接 2. 模型 ID 是否正确 3. Token 是否有效"; exit 1)
	@echo "下载完成！模型已保存在 ./qwen3_merged"

server:
	python -m vllm.entrypoints.openai.api_server \
	--model /workspace/LLM-Agent/fine-turning/minimind-chat/qwen3_merged \
	--served-model-name qwen-ft \
	--trust-remote-code \
	--gpu-memory-utilization 0.9 \
	--port 8000

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
	uv run kaggle kernels pull -p kaggle
	@echo "拉取完成！"

.PHONY: help setup train test clean push-model kinit kpush kpull