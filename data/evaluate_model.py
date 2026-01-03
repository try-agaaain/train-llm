"""
模型评估脚本
用训练好的模型对训练集中的问题进行回答，生成 qa_dataset_padding.json
"""

import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from pathlib import Path
from tqdm import tqdm
import sys

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))


class ModelEvaluator:
    def __init__(self, model_path: str, max_new_tokens: int = 512):
        """
        初始化模型评估器
        
        Args:
            model_path: 微调后的模型路径
            max_new_tokens: 生成的最大token数
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        
        print(f"📦 正在加载模型: {model_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        print("✅ 模型加载完成")
    
    def format_prompt(self, question: str) -> str:
        """格式化为 Qwen 对话格式"""
        prompt = f"""<|im_start|>system
你是一个专业的计算机网络知识助手,请根据提供的信息准确回答问题。<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def generate_answer(self, question: str) -> str:
        """
        对问题生成答案
        
        Args:
            question: 输入问题
            
        Returns:
            str: 模型生成的答案
        """
        prompt = self.format_prompt(question)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 提取助手回答部分
        try:
            answer = response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
        except:
            answer = response
        
        return answer
    
    def evaluate_dataset(self, qa_dataset_path: str, output_path: str):
        """
        评估整个数据集
        
        Args:
            qa_dataset_path: 训练集QA数据路径
            output_path: 输出文件路径
        """
        print(f"\n📂 正在加载数据集: {qa_dataset_path}")
        
        # 加载数据集
        with open(qa_dataset_path, 'r', encoding='utf-8') as f:
            qa_dataset = json.load(f)
        
        print(f"📊 数据集大小: {len(qa_dataset)} 条")
        
        # 评估结果列表
        results = []
        
        # 遍历数据集
        for idx, item in enumerate(tqdm(qa_dataset, desc="评估中")):
            question = item.get('question', '')
            expected_answer = item.get('answer', '')
            
            if not question:
                continue
            
            # 生成答案
            try:
                generated_answer = self.generate_answer(question)
            except Exception as e:
                print(f"\n⚠️ 生成答案时出错 (问题 {idx}): {e}")
                generated_answer = ""
            
            # 保存结果
            result = {
                "id": idx,
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer
            }
            results.append(result)
        
        # 保存结果
        print(f"\n💾 保存结果到: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 评估完成! 共评估 {len(results)} 条数据")
        
        return results


def main():
    """主函数"""
    # 配置参数
    model_path = "../qwen3_finetuned"  # 微调后的模型路径
    qa_dataset_path = "../data/qa_dataset.json"  # 训练集路径
    output_path = "../data/qa_dataset_padding.json"  # 输出路径
    
    # 检查路径
    model_path = Path(__file__).parent / model_path
    qa_dataset_path = Path(__file__).parent.parent / "data" / "qa_dataset.json"
    output_path = Path(__file__).parent.parent / "data" / "qa_dataset_padding.json"
    
    if not model_path.exists():
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    if not qa_dataset_path.exists():
        print(f"❌ 数据集路径不存在: {qa_dataset_path}")
        return
    
    # 初始化评估器
    evaluator = ModelEvaluator(
        model_path=str(model_path),
        max_new_tokens=512
    )
    
    # 评估数据集
    evaluator.evaluate_dataset(
        qa_dataset_path=str(qa_dataset_path),
        output_path=str(output_path)
    )


if __name__ == "__main__":
    main()
