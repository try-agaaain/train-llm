import json
import re
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel, Field
import json_repair

# --- 1. 定义输出结构 ---
class QAUnit(BaseModel):
    question: str = Field(description="基于content内容提出的具体问题")
    content: str = Field(description="【当前文本】的一个内容块，并尽量包含当前文本块的信息且能独立理解")

class Sentence(BaseModel):
    sentence: str = Field(description="句子内容")
    score: float = Field(description="该句子作为结束句子的合理性评分，分数越高表示越合理")
class ProcessingResult(BaseModel):
    content_blocks: List[QAUnit] = Field(description="提取的2-4个内容块")
    optimal_split: List[Sentence] = Field(description="【当前文本】最佳的结尾句子，用于精确边界调整")

# --- 2. 优化的提示词模板 ---
def get_system_prompt() -> str:
    return """
你是一个专业文本内容整理和文本边界分割专家。
"""

def get_human_prompt(prev_text: str, current_text: str, next_text: str) -> str:

    return f"""
## 【当前文本】
【当前文本】的内容如下：
```txt
{current_text[:-256]}
```

## 任务1：边界优化

已知上述文本的【后续内容】为：
```txt
{current_text[-256:]}{next_text[:256]}
```

请从【后续内容】中选择三句话作为【当前文本】的最佳结束句子（optimal_split）。并分别为这句话打分，评估他们作为结束句子的合理性。得分最高的optimal_split将作为【当前文本】的结束句子。而这个结束句子之后的内容将被合并到下一个文本块中。
注意：optimal_split必须是来自【后续内容】的原始句子。

## 任务2：文本整理
要求：
在得到optimal_split后，将【当前文本】到optimal_split之间的内容拆分为两到三个内容块，这几个内容块覆盖了原始文本的全部内容。随后针对每个内容块生成一个问题。
- 内容块：完整无需依赖问题或上下文，且必须覆盖【当前处理文本】的全部信息，逻辑完整，易于理解
- 覆盖度：“内容块”应覆盖【当前处理文本】的全部内容！！！
- 你的首要任务是确保切分的内容块完整且覆盖所有信息，而不是仅仅提取部分信息。


"""

# --- 3. 核心处理逻辑 ---
class BookProcessor:
    def __init__(self, 
                 provider: str = "openai",
                 model: str = "qwen-plus",
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 api_key: str = "sk-899c96c9f5b342388255efe5f3ded468", 
                 temperature: float = 0):
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
        
        # 使用结构化输出（langchain v1.0 方式）
        self.structured_llm = self.llm.with_structured_output(ProcessingResult)
        
        # 初始化token统计变量
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # 使用langchain的递归文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=0,  # 重叠部分保证上下文连贯
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
    
    def _load_existing_dataset(self, output_file: str) -> tuple[List[Dict], int]:
        """加载已有的数据集，返回数据集和下一个要处理的chunk索引"""
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            entries = data.get('entries', [])
            if entries:
                # 找出已处理的最大chunk_id
                max_chunk_id = max(entry['chunk_id'] for entry in entries)
                next_chunk_idx = max_chunk_id + 1
                
                # 恢复token统计信息
                token_usage = data.get('statistics', {}).get('token_usage', {})
                self.total_prompt_tokens = token_usage.get('prompt_tokens', 0)
                self.total_completion_tokens = token_usage.get('completion_tokens', 0)
                self.total_tokens = token_usage.get('total_tokens', 0)
                self.total_cost = token_usage.get('total_cost', 0.0)
                
                print(f"📂 加载已有数据集: {len(entries)} 个块已处理")
                print(f"   - 将从 chunk {next_chunk_idx} 继续处理")
                print(f"   - 已累计 Token: {self.total_tokens}")
                print(f"   - 已累计成本: ${self.total_cost:.4f}")
                
                return entries, next_chunk_idx
            else:
                return [], 0
        except FileNotFoundError:
            print(f"📝 未找到已有数据集，将从头开始处理")
            return [], 0
        except Exception as e:
            print(f"⚠️  加载数据集出错: {e}，将从头开始处理")
            return [], 0
        
    def process_book_to_dataset(
        self, 
        file_path: str, 
        output_file: str
    ):
        """
        处理书籍文件生成训练数据集，支持智能边界调整和断点续传
        
        Args:
            file_path: 输入文件路径
            output_file: 输出文件路径
        """
        # 尝试加载已有数据集
        dataset, start_chunk_idx = self._load_existing_dataset(output_file)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        print(f"📖 文件加载完成，总长度: {len(full_text)} 字符")
        
        # 使用langchain初步分割文本
        initial_chunks = self.text_splitter.split_text(full_text)
        print(f"✂️  初步分割完成，共 {len(initial_chunks)} 个块")
        
        chunk_idx = start_chunk_idx
        i = start_chunk_idx
        
        while i < len(initial_chunks):
            current_chunk = initial_chunks[i]
            
            if not current_chunk.strip():
                i += 1
                continue
            
            # 获取上下文
            prev_text = initial_chunks[i-1] if i > 0 else "[文档开头]"
            next_text = initial_chunks[i+1] if i < len(initial_chunks) - 1 else "[文档结尾]"
            
            # LLM处理
            try:
                result, token_stats = self._process_chunk(
                    current_text=current_chunk,
                    prev_text=prev_text,
                    next_text=next_text
                )
                
                # 根据optimal_split调整文本边界
                optimal_split = result.get('optimal_split', [])
                # 从字典列表中找出评分最高的句子
                if optimal_split:
                    optimal_split = max(optimal_split, key=lambda x: x.score).get("sentence")
                else:
                    optimal_split = None
                adjusted_text = current_chunk
                
                if optimal_split and optimal_split in current_chunk:
                    # 找到optimal_split在current_chunk中的位置
                    end_pos = current_chunk.find(optimal_split) + len(optimal_split)
                    adjusted_text = current_chunk[:end_pos]
                    
                    # 将剩余部分合并到下一个chunk
                    remaining = current_chunk[end_pos:].strip()
                    if remaining and i < len(initial_chunks) - 1:
                        initial_chunks[i+1] = remaining + " " + initial_chunks[i+1]
                    
                elif optimal_split and i < len(initial_chunks) - 1:
                    # optimal_split可能包含了下一段的开头
                    combined = current_chunk + " " + initial_chunks[i+1]
                    if optimal_split in combined:
                        end_pos = combined.find(optimal_split) + len(optimal_split)
                        adjusted_text = combined[:end_pos]
                        # 更新下一个chunk
                        initial_chunks[i+1] = combined[end_pos:].strip()

                qa_pairs = result['content_blocks']
                
                if not qa_pairs:
                    print(f"⚠️  块 {chunk_idx} 未提取到有效QA，跳过")
                    i += 1
                    continue
                
                # 保存结果
                entry = {
                    "chunk_id": chunk_idx,
                    "source_text": adjusted_text,
                    "qa_pairs": qa_pairs,
                    "metadata": {
                        "chunk_length": len(adjusted_text),
                        "boundary_adjusted": adjusted_text != current_chunk,
                        "original_chunk": current_chunk,
                        "token_usage": token_stats
                    }
                }
                dataset.append(entry)
                chunk_idx += 1
                
                # 立即保存数据集（增量保存）
                self._save_dataset(dataset, output_file)
                
                progress = (i + 1) / len(initial_chunks) * 100
                print(f"✓ 已处理: {i + 1}/{len(initial_chunks)} ({progress:.1f}%) | 已保存: {chunk_idx} | QA数: {len(qa_pairs)} | Prompt Tokens: {token_stats['prompt_tokens']} | Completion Tokens: {token_stats['completion_tokens']} | Total Cost: ${token_stats['total_cost']:.4f}")
                
            except Exception as e:
                print(f"❌ 处理出错 (chunk {i}): {e}")
            
            i += 1
        
        # 最后再保存一次确保数据完整（虽然每次都保存了，但这里再保存一次作为最终确认）
        self._save_dataset(dataset, output_file)
        print(f"\n🎉 处理完成！")
        print(f"   - 总块数: {len(dataset)}")
        print(f"   - 总QA数: {sum(len(d['qa_pairs']) for d in dataset)}")
        print(f"   - 输出文件: {output_file}")
        print(f"\n📊 Token 使用统计：")
        print(f"   - 总 Token 数: {self.total_tokens}")
        print(f"   - 提示 Token 数: {self.total_prompt_tokens}")
        print(f"   - 完成 Token 数: {self.total_completion_tokens}")
        print(f"   - 预估消耗金额: ${self.total_cost:.4f}")
    
    def _fix_json(self, raw_output: str) -> Dict:
        """尝试修复非法 JSON"""
        try:
            # 使用 json_repair 修复 JSON
            repaired_json = json_repair.repair_json(raw_output)
            print("🔧 JSON 修复成功")
            return json.loads(repaired_json)
        except Exception as e:
            print("❌ JSON 修复失败，返回原始错误")
            raise e

    def _process_chunk(self, current_text: str, prev_text: str, next_text: str) -> tuple[Dict, Dict]:
        """调用LLM处理单个文本块，提供上下文信息，并返回token统计信息"""
        
        system_prompt = get_system_prompt()
        human_content = get_human_prompt(
            prev_text=prev_text,
            current_text=current_text,
            next_text=next_text
        )
        
        # 显式构建 messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ]
        
        token_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }
        
        try:
            with get_openai_callback() as cb:
                # 使用结构化输出直接调用
                response = self.structured_llm.invoke(messages)
                
                # 记录token统计信息
                token_stats["prompt_tokens"] = cb.prompt_tokens
                token_stats["completion_tokens"] = cb.completion_tokens
                token_stats["total_tokens"] = cb.total_tokens
                token_stats["total_cost"] = cb.total_cost
                
                # 累加到总计
                self.total_prompt_tokens += cb.prompt_tokens
                self.total_completion_tokens += cb.completion_tokens
                self.total_tokens += cb.total_tokens
                self.total_cost += cb.total_cost
                
            # 将 Pydantic 对象转换为字典
            return response.model_dump(), token_stats
        except Exception as e:
            # 其他类型的错误直接抛出
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"❌ {error_type}: {error_msg}")
            raise
    
    def _save_dataset(self, dataset: List[Dict], output_file: str):
        """保存为JSON格式，包含token统计信息"""
        output_data = {
            "entries": dataset,
            "statistics": {
                "total_chunks": len(dataset),
                "total_qa_pairs": sum(len(d['qa_pairs']) for d in dataset),
                "token_usage": {
                    "total_tokens": self.total_tokens,
                    "prompt_tokens": self.total_prompt_tokens,
                    "completion_tokens": self.total_completion_tokens,
                    "total_cost": self.total_cost
                }
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

# --- 4. 使用示例 ---
if __name__ == "__main__":
    # 示例1: 使用OpenAI兼容接口 (如qwen-plus)
    processor_openai = BookProcessor(
        provider="openai",
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-899c96c9f5b342388255efe5f3ded468",
        temperature=0
    )
    
    # 示例2: 使用GLM (智谱AI)
    processor_glm = BookProcessor(
        provider="glm",
        model="glm-4-flash",
        api_key="f256244ea8754fd290f260d4908c5062.BmhL8o2oneESP67X",
        temperature=0
    )
    
    # 示例3: 使用Gemini (Google AI)
    processor_gemini = BookProcessor(
        provider="gemini",
        model="gemini-2.5-flash",
        api_key="AIzaSyCZMvCygYo_EVvfbC1R7unG29zFd7R_IB0",
        temperature=0
    )
    
    # 选择使用的处理器
    processor = processor_openai
    
    # 处理书籍
    processor.process_book_to_dataset(
        file_path=r"/workspace/LLM-Agent/fine-turning/minimind-chat/train/MinerU_markdown_计算机网络-第7版-谢希仁_20251223084758_2003266285337022464.md",
        output_file="train_dataset.jsonl"
    )
    
    # 批量处理多个文件
    # for book in ["book1.txt", "book2.txt", "book3.txt"]:
    #     processor.process_book_to_dataset(book, f"{book}_dataset.jsonl")