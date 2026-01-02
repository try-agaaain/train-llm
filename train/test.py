"""
Qwen3 0.6B 模型对比测试脚本
交互式对比训练前后模型的回答效果
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ==================== 配置参数 ====================
class Config:
    # 原始模型路径
    original_model_name = "Qwen/Qwen3-0.6B"
    
    # 微调后模型路径
    finetuned_model_path = "./qwen3_finetuned"
    
    # 生成参数
    max_new_tokens = 512
    temperature = 0.01
    top_p = 0.1

# ==================== 模型加载 ====================
def load_models(config):
    """加载原始模型和微调后的模型"""
    
    # 加载原始模型
    logger.info(f"加载原始模型: {config.original_model_name}")
    original_tokenizer = AutoTokenizer.from_pretrained(
        config.original_model_name,
        trust_remote_code=True
    )
    original_model = AutoModelForCausalLM.from_pretrained(
        config.original_model_name,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # 加载微调后的模型
    logger.info(f"加载微调后模型: {config.finetuned_model_path}")
    finetuned_tokenizer = AutoTokenizer.from_pretrained(
        config.finetuned_model_path,
        trust_remote_code=True
    )
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        config.finetuned_model_path,
        trust_remote_code=True,
        device_map="auto"
    )
    
    logger.info("模型加载完成!")
    return (original_model, original_tokenizer), (finetuned_model, finetuned_tokenizer)

# ==================== 推理函数 ====================
def generate_answer(model, tokenizer, question, config):
    """生成回答"""
    # 构建提示
    prompt = f"""<|im_start|>system
你是一个专业的计算机网络知识助手,请根据提供的信息准确回答问题。<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    
    # 生成回答
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取助手回答部分
    answer = response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    
    return answer

# ==================== 交互式测试 ====================
def interactive_test(config):
    """交互式测试函数"""
    print("\n" + "="*80)
    print("Qwen3 模型对比测试系统")
    print("="*80)
    print("功能: 对比训练前后模型的回答效果")
    print("输入 'quit' 或 'exit' 退出程序\n")
    
    # 加载模型
    (original_model, original_tokenizer), (finetuned_model, finetuned_tokenizer) = load_models(config)
    
    # 交互循环
    while True:
        # 获取用户输入
        print("-" * 80)
        question = input("\n请输入您的问题: ").strip()
        
        # 检查退出命令
        if question.lower() in ['quit', 'exit', '退出']:
            print("\n感谢使用,再见!")
            break
        
        # 跳过空输入
        if not question:
            print("输入为空,请重新输入!")
            continue
        
        print("\n" + "="*80)
        print(f"问题: {question}")
        print("="*80)
        
        # 生成原始模型回答
        print("\n【训练前模型回答】")
        print("-" * 80)
        try:
            original_answer = generate_answer(original_model, original_tokenizer, question, config)
            print(original_answer)
        except Exception as e:
            print(f"生成失败: {str(e)}")
        
        # 生成微调后模型回答
        print("\n【训练后模型回答】")
        print("-" * 80)
        try:
            finetuned_answer = generate_answer(finetuned_model, finetuned_tokenizer, question, config)
            print(finetuned_answer)
        except Exception as e:
            print(f"生成失败: {str(e)}")
        
        print("\n")

# ==================== 主函数 ====================
if __name__ == "__main__":
    config = Config()
    
    try:
        interactive_test(config)
    except KeyboardInterrupt:
        print("\n\n程序被用户中断,退出...")
    except Exception as e:
        logger.error(f"程序出错: {str(e)}", exc_info=True)