"""
数据集增强脚本
使用 qwen-plus 评估模型回答的准确性，分析错误原因，生成新的QA对
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from tqdm import tqdm

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_env, get_api_key, get_base_url

# 加载环境变量
load_env(__file__)


# ==================== 定义输出结构 ====================
class QAPair(BaseModel):
    question: str = Field(description="新生成的问题")
    answer: str = Field(description="新生成的答案")


class EvaluationResult(BaseModel):
    is_acceptable: bool = Field(description="问题本身是否可接受（True表示问题有效，False表示问题本身就有问题）")
    is_correct: bool = Field(description="模型回答是否正确准确")
    error_analysis: str = Field(description="如果回答不正确，分析错误原因和缺少的信息")
    new_qa_pairs: List[QAPair] = Field(description="针对错误或不足生成的1-3个新QA对，用于补充数据集")


# ==================== 提示词模板 ====================
def get_evaluation_system_prompt() -> str:
    return """你是一个专业的问答质量评估专家。
你的任务是评估模型对问题的回答质量，并在回答不准确时分析原因并生成补充的QA对。"""


def get_evaluation_human_prompt(question: str, expected_answer: str, generated_answer: str) -> str:
    return f"""## 任务：评估问答质量并生成补充数据

### 【原始问题】
{question}

### 【期望答案】
{expected_answer}

### 【模型生成的答案】
{generated_answer}

## 评估要求

### 1. 问题可接受性判断 (is_acceptable)
首先判断问题本身是否有效：
- ✅ 可接受：问题清晰、独立、不依赖上下文即可理解和回答
- ❌ 不可接受：
  - 依赖上下文才能理解（如"这个是什么？"、"刚才提到的是？"）
  - 问题本身描述错误或有歧义
  - 包含"如图"、"上表"等视觉引用但缺少具体信息

### 2. 回答正确性判断 (is_correct)
如果问题可接受，评估模型的回答：
- ✅ 正确：回答准确、完整、与期望答案一致或表达相同意思
- ❌ 错误：
  - 回答不准确或有重要遗漏
  - 回答偏离主题
  - 回答包含错误信息

### 3. 错误原因分析 (error_analysis)
如果回答不正确或问题不可接受，分析原因：
- 缺少哪些关键知识点或信息
- 对哪些概念理解有偏差
- 数据集中缺少哪方面的训练样本
- 如果问题不可接受，说明问题本身的缺陷

### 4. 生成新QA对 (new_qa_pairs)
**仅当问题可接受且回答不正确时**，生成1-3个新的QA对来补充数据集：
- 针对缺失的知识点生成问题
- 确保新QA对可以帮助模型学习到正确的知识
- 问题必须清晰独立，答案必须准确完整
- **如果问题不可接受(is_acceptable=False)，则new_qa_pairs必须为空列表**

## 示例

### 示例1：问题可接受但回答错误
- 问题：TCP三次握手的过程是什么？
- 期望答案：客户端发送SYN，服务器返回SYN-ACK，客户端发送ACK
- 模型回答：TCP通过三次握手建立连接（缺少具体步骤）

评估结果：
```json
{{
  "is_acceptable": true,
  "is_correct": false,
  "error_analysis": "模型回答过于笼统，缺少三次握手的具体步骤和每一步的详细内容。数据集中可能缺少对TCP握手过程每个步骤的详细描述。",
  "new_qa_pairs": [
    {{
      "question": "TCP三次握手的第一步是什么？",
      "answer": "客户端发送一个SYN（同步）报文段到服务器，表示希望建立连接。"
    }},
    {{
      "question": "TCP三次握手的第二步服务器如何响应？",
      "answer": "服务器返回一个SYN-ACK报文段，确认收到客户端的SYN并也发送自己的SYN。"
    }},
    {{
      "question": "TCP三次握手的第三步是什么？",
      "answer": "客户端发送ACK报文段确认收到服务器的SYN-ACK，连接建立完成。"
    }}
  ]
}}
```

### 示例2：问题不可接受
- 问题：如图所示的网络拓扑中，A和B之间的延迟是多少？
- 模型回答：根据图示，延迟是10ms

评估结果：
```json
{{
  "is_acceptable": false,
  "is_correct": false,
  "error_analysis": "问题依赖于'如图所示'的视觉信息，但数据集中没有图片，问题本身缺少必要的上下文信息，无法独立理解和回答。",
  "new_qa_pairs": []
}}
```

### 示例3：回答正确
- 问题：什么是IP地址？
- 期望答案：IP地址是互联网协议地址，用于标识网络中的设备
- 模型回答：IP地址是互联网协议地址，每个连接到互联网的设备都有一个唯一的IP地址用于标识

评估结果：
```json
{{
  "is_acceptable": true,
  "is_correct": true,
  "error_analysis": "",
  "new_qa_pairs": []
}}
```

## 注意事项
- **关键规则**：只有当is_acceptable=True且is_correct=False时，才生成new_qa_pairs
- 新生成的QA对质量要高，确保答案准确、完整
- 聚焦于补充模型缺失的知识点

请开始评估："""


# ==================== 数据集增强器 ====================
class DatasetAugmenter:
    def __init__(self, 
                 model: str = "qwen-plus",
                 temperature: float = 0.1):
        """
        初始化数据集增强器
        
        Args:
            model: 使用的LLM模型
            temperature: 温度参数
        """
        # 从环境变量加载API配置
        api_key = get_api_key("QWEN")
        base_url = get_base_url("QWEN")
        
        self.llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature
        )
        
        # 创建结构化输出
        self.eval_llm = self.llm.with_structured_output(EvaluationResult)
        
        print(f"✅ 初始化完成，使用模型: {model}")
    
    def evaluate_qa(self, question: str, expected_answer: str, generated_answer: str) -> EvaluationResult:
        """
        评估单个QA对
        
        Args:
            question: 问题
            expected_answer: 期望答案
            generated_answer: 模型生成的答案
            
        Returns:
            EvaluationResult: 评估结果
        """
        messages = [
            SystemMessage(content=get_evaluation_system_prompt()),
            HumanMessage(content=get_evaluation_human_prompt(question, expected_answer, generated_answer))
        ]
        
        try:
            result = self.eval_llm.invoke(messages)
            return result
        except Exception as e:
            print(f"\n❌ 评估出错: {e}")
            # 返回默认结果
            return EvaluationResult(
                is_acceptable=True,
                is_correct=True,
                error_analysis="",
                new_qa_pairs=[]
            )
    
    def augment_dataset(self, padding_file: str, output_file: str):
        """
        增强数据集
        
        Args:
            padding_file: 模型评估结果文件 (qa_dataset_padding.json)
            output_file: 增强后的数据集输出文件
        """
        print(f"\n📂 正在加载评估结果: {padding_file}")
        
        # 加载评估结果
        with open(padding_file, 'r', encoding='utf-8') as f:
            padding_data = json.load(f)
        
        print(f"📊 数据集大小: {len(padding_data)} 条")
        
        # 统计信息
        stats = {
            "total": len(padding_data),
            "acceptable": 0,
            "correct": 0,
            "incorrect": 0,
            "unacceptable": 0,
            "new_qa_count": 0
        }
        
        # 增强结果列表
        augmented_results = []
        new_qa_pairs = []
        
        # 遍历评估结果
        for idx, item in enumerate(tqdm(padding_data, desc="增强数据集")):
            question = item.get('question', '')
            expected_answer = item.get('expected_answer', '')
            generated_answer = item.get('generated_answer', '')
            
            if not question:
                continue
            
            # 使用qwen-plus评估
            try:
                eval_result = self.evaluate_qa(question, expected_answer, generated_answer)
            except Exception as e:
                print(f"\n⚠️ 评估问题 {idx} 时出错: {e}")
                continue
            
            # 更新统计
            if eval_result.is_acceptable:
                stats["acceptable"] += 1
                if eval_result.is_correct:
                    stats["correct"] += 1
                else:
                    stats["incorrect"] += 1
            else:
                stats["unacceptable"] += 1
            
            # 保存评估结果
            result = {
                "id": idx,
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "is_acceptable": eval_result.is_acceptable,
                "is_correct": eval_result.is_correct,
                "error_analysis": eval_result.error_analysis,
                "new_qa_pairs": [qa.model_dump() for qa in eval_result.new_qa_pairs]
            }
            augmented_results.append(result)
            
            # 收集新生成的QA对（只收集可接受问题的补充QA）
            if eval_result.is_acceptable and eval_result.new_qa_pairs:
                for qa in eval_result.new_qa_pairs:
                    new_qa_pairs.append({
                        "question": qa.question,
                        "answer": qa.answer,
                        "source": f"augmented_from_question_{idx}"
                    })
                    stats["new_qa_count"] += 1
        
        # 保存增强结果
        output_data = {
            "augmented_results": augmented_results,
            "new_qa_pairs": new_qa_pairs,
            "statistics": stats
        }
        
        print(f"\n💾 保存增强结果到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # 打印统计信息
        print("\n" + "="*50)
        print("📊 增强统计信息:")
        print(f"   总问题数: {stats['total']}")
        print(f"   可接受问题: {stats['acceptable']} ({stats['acceptable']/stats['total']*100:.1f}%)")
        print(f"   不可接受问题: {stats['unacceptable']} ({stats['unacceptable']/stats['total']*100:.1f}%)")
        print(f"   回答正确: {stats['correct']} ({stats['correct']/stats['acceptable']*100:.1f}% of acceptable)")
        print(f"   回答错误: {stats['incorrect']} ({stats['incorrect']/stats['acceptable']*100:.1f}% of acceptable)")
        print(f"   新生成QA对: {stats['new_qa_count']}")
        print("="*50)
        
        return output_data


def main():
    """主函数"""
    # 配置路径
    padding_file = Path(__file__).parent / "qa_dataset_padding.json"
    output_file = Path(__file__).parent / "qa_dataset_augmented.json"
    
    if not padding_file.exists():
        print(f"❌ 评估结果文件不存在: {padding_file}")
        print("请先运行 evaluate_model.py 生成评估结果")
        return
    
    # 初始化增强器
    augmenter = DatasetAugmenter(
        model="qwen-plus",
        temperature=0.1
    )
    
    # 增强数据集
    augmenter.augment_dataset(
        padding_file=str(padding_file),
        output_file=str(output_file)
    )
    
    print("\n✅ 数据集增强完成!")


if __name__ == "__main__":
    main()
