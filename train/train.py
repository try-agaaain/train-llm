"""
Qwen3 0.6B 模型微调训练脚本
使用 LoRA 进行高效微调,支持自定义问答数据集
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import logging
import pathlib

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ==================== 配置参数 ====================
class Config:
    # 模型配置
    # model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Qwen3 0.6B 模型
    model_name = "Qwen/Qwen3-0.6B"
    # model_name = "Qwen/Qwen3-0.6B-FP8"
    # model_name = "rd211/Qwen3-0.6B-Instruct"
    
    # 数据配置
    train_data_path = pathlib.Path(__file__).parent.parent / "artifacts" / "dataset" / "qa_dataset.json"    
    max_length = 512
    
    # LoRA 配置
    lora_r = 8  # LoRA 秩
    lora_alpha = 16  # LoRA alpha
    lora_dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 目标层
    # target_modules = ["o_proj"]  # 目标层
    
    # 训练配置
    output_dir = str(pathlib.Path(__file__).parent.parent / "artifacts" / "models" / "qwen3_finetuned")
    num_epochs = 5
    batch_size = 4  # 减小批次大小
    gradient_accumulation_steps = 16  # 增加梯度累积步数保持有效批次
    learning_rate = 2e-4
    warmup_steps = 100
    logging_steps = 10
    save_steps = 500
    
    # 硬件配置
    fp16 = False  # 禁用fp16避免mask类型问题
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 数据处理 ====================
def load_and_prepare_data(file_path):
    """加载并准备训练数据"""
    logger.info(f"加载数据: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为 Hugging Face Dataset
    dataset = Dataset.from_list(data)
    logger.info(f"数据集大小: {len(dataset)}")
    
    return dataset

def format_prompt(example):
    """格式化为 Qwen 对话格式"""
    prompt = f"""<|im_start|>system
你是一个专业的计算机网络知识助手,请根据提供的信息准确回答问题。<|im_end|>
<|im_start|>user
{example['question']}<|im_end|>
<|im_start|>assistant
{example['answer']}<|im_end|>"""
    return prompt

def tokenize_function(example, tokenizer, max_length):
    """对数据进行 tokenization"""
    prompt = format_prompt(example)
    
    # Tokenize
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    
    # 设置 labels (用于计算 loss)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# ==================== 模型加载与配置 ====================
def load_model_and_tokenizer(config):
    """加载预训练模型和 tokenizer"""
    logger.info(f"加载模型: {config.model_name}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # 确保有 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        # low_cpu_mem_usage=True
        low_cpu_mem_usage=False
        # 移除 attn_implementation，让库自动选择
    )
    
    # 将模型移到GPU
    if config.device == "cuda":
        model = model.cuda()
    
    # 配置 LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

# ==================== 训练 ====================
def train(config):
    """主训练函数"""
    # 加载数据
    dataset = load_and_prepare_data(config.train_data_path)
    
    # 加载模型和 tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Tokenize 数据
    logger.info("处理数据...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(
            examples, 
            tokenizer, 
            config.max_length
        ),
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        fp16=config.fp16,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",  # 不使用 wandb 等
        remove_unused_columns=False,
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 保存模型
    logger.info(f"保存模型到: {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    logger.info("训练完成!")

# ==================== 推理测试 ====================
def test_model(config, test_question):
    """测试微调后的模型"""
    logger.info("加载微调后的模型进行测试...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.output_dir,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.output_dir,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # 构建测试提示
    prompt = f"""<|im_start|>system
你是一个专业的计算机网络知识助手,请根据提供的信息准确回答问题。<|im_end|>
<|im_start|>user
{test_question}<|im_end|>
<|im_start|>assistant
"""
    
    # 生成回答
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取助手回答部分
    answer = response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    
    print(f"\n问题: {test_question}")
    print(f"\n回答: {answer}\n")

# ==================== 主函数 ====================
if __name__ == "__main__":
    config = Config()
    
    # 训练模型
    train(config)
    
    # 测试模型
    # test_question = "为什么HLR和VLR需要具有高可靠性和短响应时间？"
    test_question = "为什么无线局域网不能使用CSMA/CD协议中的碰撞检测功能？"
    test_model(config, test_question)