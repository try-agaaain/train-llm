import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel, Field

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_env, get_api_key, get_base_url

# 加载环境变量
load_env(__file__)

# --- 1. 定义输出结构 ---
class QAPair(BaseModel):
    question: str = Field(description="基于文本块生成的针对性问题。严禁包含'如图所示'、'上表'、'文中'等上下文依赖的表达。")
    answer: str = Field(description="基于文本块对问题的详细回答。必须是上下文无关的（Self-contained），不要使用'根据文本'、'文中提到'等前缀，直接陈述事实。")

# --- 2. 提示词模板 ---
def get_system_prompt() -> str:
    return """你是一个专业的教育数据集生成专家。
你的任务是根据给定的文本片段，生成高质量的问答对（QA Pair）。
目标是让模型能够通过这个问题和答案学习到文本中的知识点。"""

def get_human_prompt(text: str) -> str:
    return f"""## 任务：基于文本生成独立的问答对

### 【参考文本】
```txt
{text}
```

## 要求

### 1. 问题生成 (Question)
- 针对文本中的核心信息提一个具体的问题。
- **去上下文（关键）**：问题必须是独立的。
  - ❌ 错误：这个图展示了什么？ / 文中提到的第三点是什么？ / 表格中的数据说明了什么？
  - ✅ 正确：TCP协议的三次握手过程是怎样的？ / 什么是操作系统的虚拟化技术？

### 2. 答案生成 (Answer)
- 基于【参考文本】的内容回答刚才提出的问题。
- **去引用（关键）**：答案必须是客观陈述，不依赖于文本引用，必须是自包含的，逻辑清晰的完整回答，并尽可能包含详细信息。
  - ❌ 错误：根据这段文字... / 作者在文中指出... / 如图1-2所示...
  - ✅ 正确：TCP三次握手是指... / 虚拟化技术通过...
- **图表转化**：如果文本包含"如图所示"等描述，请将视觉信息转化为文字描述，或者在答案中忽略视觉引用的部分，只保留知识性内容。

请生成一个JSON对象，包含 `question` 和 `answer`。"""

# --- 3. 核心处理逻辑 ---
class QAGenerator:
    def __init__(self, 
                 provider: str = "openai",
                 model: str = "qwen-plus",
                 base_url: str = None,
                 api_key: str = None, 
                 temperature: float = 0.1):
        
        # 如果没有提供API key，从环境变量加载
        if api_key is None:
            if provider == "openai":
                api_key = get_api_key("QWEN")
            elif provider == "glm":
                api_key = get_api_key("GLM")
            elif provider == "gemini":
                api_key = get_api_key("GEMINI")
        
        # 如果没有提供base_url，从环境变量或默认值加载
        if base_url is None and provider == "openai":
            base_url = get_base_url("QWEN")
        
        if provider == "openai":
            self.llm = ChatOpenAI(
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature
            )
        elif provider == "glm":
            from langchain_community.chat_models import ChatZhipuAI
            self.llm = ChatZhipuAI(
                model=model,
                api_key=api_key,
                temperature=temperature
            )
        elif provider == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                api_key=api_key,
                temperature=temperature
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # 结构化输出
        self.qa_llm = self.llm.with_structured_output(QAPair)
        
        # 统计信息
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # 文本切分器：按256字符切分，保留少量重叠以防切断关键词
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,
            chunk_overlap=20,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )

    def _load_existing_dataset(self, output_file: str) -> Tuple[List[Dict], int]:
        """加载断点"""
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    entries = data.get('entries', [])
                    # 恢复统计信息
                    stats = data.get('statistics', {}).get('token_usage', {})
                    self.total_tokens = stats.get('total_tokens', 0)
                    self.total_cost = stats.get('total_cost', 0.0)
                    
                    print(f"📂 加载已有进度: {len(entries)} 条数据")
                    return entries, len(entries)
            except Exception as e:
                print(f"⚠️ 读取文件失败: {e}，将重新开始")
        return [], 0

    def _save_dataset(self, dataset: List[Dict], output_file: str):
        """保存数据"""
        output_data = {
            "entries": dataset,
            "statistics": {
                "count": len(dataset),
                "token_usage": {
                    "total_tokens": self.total_tokens,
                    "total_cost": self.total_cost
                }
            }
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    def generate_qa(self, text_chunk: str) -> Tuple[QAPair, Dict]:
        """调用LLM生成QA"""
        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=get_human_prompt(text_chunk))
        ]
        
        token_stats = {}
        
        try:
            with get_openai_callback() as cb:
                result = self.qa_llm.invoke(messages)
                
                token_stats = {
                    "total_tokens": cb.total_tokens,
                    "cost": cb.total_cost
                }
                self.total_tokens += cb.total_tokens
                self.total_cost += cb.total_cost
                
            return result, token_stats
        except Exception as e:
            print(f"❌ LLM调用错误: {e}")
            return None, {}

    def process_file(self, file_path: str, output_file: str):
        """主处理流程"""
        dataset, start_idx = self._load_existing_dataset(output_file)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
            
        # 1. 切分文本
        chunks = self.text_splitter.split_text(full_text)
        print(f"✂️  文本已切分为 {len(chunks)} 个块 (Chunk Size: 256)")
        
        # 2. 遍历处理
        for i in range(start_idx, len(chunks)):
            chunk = chunks[i]
            if len(chunk.strip()) < 10: # 跳过过短的块
                continue
                
            print(f"\n🚀 [{i+1}/{len(chunks)}] 处理中...")
            
            # 生成QA
            qa_result, stats = self.generate_qa(chunk)
            
            if qa_result:
                entry = {
                    "id": i,
                    "text": chunk,
                    "question": qa_result.question,
                    "answer": qa_result.answer,
                    "token_usage": stats
                }
                dataset.append(entry)
                
                print(f"   Q: {qa_result.question}")
                print(f"   A: {qa_result.answer[:50]}...")
                
                # 每5条保存一次，或者最后一条保存
                if (i + 1) % 5 == 0 or i == len(chunks) - 1:
                    self._save_dataset(dataset, output_file)
                    print(f"   💾 进度已保存")
            else:
                print("   ⚠️ 生成失败，跳过")

        print(f"\n🎉 处理完成! 总计生成 {len(dataset)} 条QA数据。")
        print(f"💰 总成本预估: ${self.total_cost:.4f}")

if __name__ == "__main__":
    # 初始化生成器（API key会自动从环境变量加载）
    generator = QAGenerator(
        provider="openai",
        model="qwen-plus", # 或 qwen-max, gpt-4o
        temperature=0.1
    )
    
    # 设置输入输出路径
    # 请根据实际情况修改文件名
    input_file = "/workspace/LLM-Agent/fine-turning/minimind-chat/dataset/MinerU_markdown_计算机网络-第7版-谢希仁_20251223084758_2003266285337022464.md"
    output_file = "qa_dataset_new.json"
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        # 尝试在当前目录下查找
        input_file = os.path.basename(input_file)
    
    if os.path.exists(input_file):
        generator.process_file(input_file, output_file)
    else:
        print(f"❌ 找不到输入文件: {input_file}")
