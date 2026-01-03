# 环境设置
VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
MODELSCOPE_USER=SoFarSoLong
MODEL_NAME=qwen3_finetuned
MODELSCOPE_TOKEN=xxx
# 默认目标
.DEFAULT_GOAL := help

# 帮助信息
help:
	@echo "可用命令:"
	@echo "  make setup       初始化开发环境"
	@echo "  make train       训练模型"
	@echo "  make test        运行测试"
	@echo "  make clean       清理生成文件"
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

# 推送模型到ModelScope
PUSH_DIR := ./qwen3_finetuned_for_upload

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
	cp ./qwen3_finetuned/{adapter_config.json,adapter_model.safetensors,merges.txt} $(PUSH_DIR)/
	# 复制 tokenizer 文件
	cp ./qwen3_finetuned/{added_tokens.json,chat_template.jinja,special_tokens_map.json,tokenizer.json,tokenizer_config.json,vocab.json} $(PUSH_DIR)/
	# 复制 README.md
	cp ./qwen3_finetuned/README.md $(PUSH_DIR)/
	@echo "上传目录准备完毕。"

pull-model:
	@echo "正在从ModelScope拉取模型..."
	uv run modelscope download \
		$(MODELSCOPE_USER)/$(MODEL_NAME) \
		--local-dir ./qwen3_finetuned \
		--repo-type model \
		--token $(MODELSCOPE_TOKEN) \
		|| (echo "拉取失败，请检查：1. modelscope是否安装 2. 环境变量是否正确设置"; exit 1)
	@echo "拉取成功！模型已保存到：./qwen3_finetuned"

.PHONY: help setup train test clean push-model
