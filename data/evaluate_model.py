"""
模型评估脚本 - 支持批处理和命令行参数

用训练好的模型对训练集中的问题进行批量回答，生成 qa_dataset_padding.json
"""

import json
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from pathlib import Path
from tqdm import tqdm
import sys
from typing import List, Dict

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from utils import get_artifacts_dir


class BatchModelEvaluator:
    def __init__(self, model_path: str, max_new_tokens: int = 512, batch_size: int = 4):
        """
        初始化批处理模型评估器
        
        Args:
            model_path: 微调后的模型路径
            max_new_tokens: 生成的最大token数
            batch_size: 批处理大小，同时处理的问题数量
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        
        print(f"📦 正在加载模型: {model_path}")
        print(f"⚙️  批处理大小: {batch_size}")
        print(f"🔢 最大生成token数: {max_new_tokens}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 设置padding token（如果没有的话）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        print("✅ 模型加载完成\n")
    
    def format_prompt(self, question: str) -> str:
        """格式化为 Qwen 对话格式"""
        prompt = f"""<|im_start|>system
你是一个专业的计算机网络知识助手,请根据提供的信息准确回答问题。<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def generate_answers_batch(self, questions: List[str]) -> List[str]:
        """
        批量生成答案
        
        Args:
            questions: 问题列表
            
        Returns:
            List[str]: 生成的答案列表
        """
        # 格式化所有问题
        prompts = [self.format_prompt(q) for q in questions]
        
        # 批量tokenize，padding到相同长度
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt",
            padding=True,  # 自动padding到batch中最长的序列
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # 批量生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,

            )
        
        # 批量解码
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        # 提取答案
        answers = []
        for response in responses:
            try:
                answer = response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
            except:
                answer = response.strip()
            answers.append(answer)
        
        return answers
    
    def evaluate_dataset(self, qa_dataset_path: str, output_path: str):
        """
        评估整个数据集（批处理版本）
        
        Args:
            qa_dataset_path: 训练集QA数据路径
            output_path: 输出文件路径
        """
        print(f"📂 正在加载数据集: {qa_dataset_path}")
        
        # 加载数据集
        with open(qa_dataset_path, 'r', encoding='utf-8') as f:
            qa_dataset = json.load(f)
        
        # 如果数据集是字典 (例如, {"id": {...}, ...} 格式), 转换为值列表以便切片
        if isinstance(qa_dataset, dict):
            qa_dataset = list(qa_dataset.get("entries"))

        
        print(f"📊 数据集大小: {len(qa_dataset)} 条\n")
        
        # 评估结果列表
        results = []
        
        # 按batch_size分批处理
        num_batches = (len(qa_dataset) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="批处理评估中"):
            # 获取当前批次的数据
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(qa_dataset))
            batch_items = qa_dataset[start_idx:end_idx]
            
            # 提取问题
            questions = [item.get('question', '') for item in batch_items]
            questions = [q for q in questions if q]  # 过滤空问题
            
            if not questions:
                continue
            
            # 批量生成答案
            try:
                generated_answers = self.generate_answers_batch(questions)
            except Exception as e:
                print(f"\n⚠️ 批次 {batch_idx} 生成答案时出错: {e}")
                # 如果批处理失败，降级为单个处理
                generated_answers = []
                for q in questions:
                    try:
                        ans = self.generate_answers_batch([q])[0]
                        generated_answers.append(ans)
                    except:
                        generated_answers.append("")
            
            # 保存结果
            for i, item in enumerate(batch_items):
                if i < len(generated_answers):
                    result = {
                        "id": start_idx + i,
                        "question": item.get('question', ''),
                        "expected_answer": item.get('answer', ''),
                        "generated_answer": generated_answers[i]
                    }
                    results.append(result)
        
        # 保存结果
        print(f"\n💾 保存结果到: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 评估完成! 共评估 {len(results)} 条数据")
        
        return results


def parse_arguments():
    pwd = Path(__file__).parent
    
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="批处理模型评估脚本 - 使用微调后的模型生成答案",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 指定模型和数据集路径
  python evaluate.py --model ./models/qwen3_finetuned --dataset ./data/qa_dataset.json
  # 指定artifacts目录
  python evaluate.py --artifacts-dir /custom/path
"""
    )
    
    parser.add_argument(
        '--artifacts-dir',
        type=str,
        default=None,
        help='Artifacts目录路径 (优先级: 参数 > 环境变量 > 默认值)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='微调后的模型路径（相对或绝对路径）'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='训练集QA数据路径（相对或绝对路径）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径（相对或绝对路径）'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='批处理大小，根据GPU显存调整 (默认: 8)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1024,
        help='生成的最大token数 (默认: 1024)'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 获取artifacts目录
    artifacts_dir = get_artifacts_dir(custom_path=args.artifacts_dir, current_file=__file__)
    
    # 设置路径（支持相对和绝对路径）
    if args.model:
        model_path = Path(args.model) if Path(args.model).is_absolute() else artifacts_dir / args.model
    else:
        model_path = artifacts_dir / "models" / "qwen3_finetuned"
    
    if args.dataset:
        qa_dataset_path = Path(args.dataset) if Path(args.dataset).is_absolute() else artifacts_dir / args.dataset
    else:
        qa_dataset_path = artifacts_dir / "dataset" / "qa_dataset.json"
    
    if args.output:
        output_path = Path(args.output) if Path(args.output).is_absolute() else artifacts_dir / args.output
    else:
        output_path = artifacts_dir / "dataset" / "qa_dataset_padding.json"
    
    print(f"📁 Artifacts目录: {artifacts_dir}")
    print(f"📁 模型路径: {model_path}")
    print(f"📁 数据集路径: {qa_dataset_path}")
    print(f"📁 输出路径: {output_path}")
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    if not qa_dataset_path.exists():
        print(f"❌ 数据集路径不存在: {qa_dataset_path}")
        return
    
    # 初始化评估器
    evaluator = BatchModelEvaluator(
        model_path=str(model_path),
        max_new_tokens=args.max_tokens,
        batch_size=args.batch_size
    )
    
    # 评估数据集
    evaluator.evaluate_dataset(
        qa_dataset_path=str(qa_dataset_path),
        output_path=str(output_path)
    )


if __name__ == "__main__":
    main()