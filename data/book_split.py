import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel, Field
import json_repair

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_env, get_api_key, get_base_url

# 加载环境变量
load_env(__file__)

# --- 1. 定义输出结构 ---
class BoundaryLine(BaseModel):
    line_number: int = Field(description="切分边界的行号（在该行之后进行切分）")
    reason: str = Field(description="选择该行作为切分边界的理由")
    score: float = Field(description="该行作为切分边界的合理性评分，分数越高表示越合理")

class BoundaryResult(BaseModel):
    optimal_splits: List[BoundaryLine] = Field(description="3个候选切分边界（行号），按评分从高到低排序")

class QAUnit(BaseModel):
    content: str = Field(description="【当前文本】的一个内容块，并尽量包含当前文本块的信息且能独立理解")
    question: str = Field(description="基于content内容提出的具体问题")

class ContentBlockResult(BaseModel):
    content_blocks: List[QAUnit] = Field(description="提取的2-4个内容块")

# --- 2. 优化的提示词模板 ---
def get_boundary_system_prompt() -> str:
    return """你是一个专业的文本边界分割专家。
你的任务是找出文本的最佳结束位置（以行号表示），确保文本在语义和结构上的完整性。"""

def get_boundary_human_prompt(numbered_lines: List[str]) -> str:
    lines_text = "\n".join(numbered_lines)
    
    return f"""## 任务：找出文本的最佳结束边界

### 【带行号的边界文本】
```
{lines_text}
```

请从上述带行号的文本中选择**3个行号**作为最佳切分边界，并为每个边界打分。

**说明**：
- 行号表示在该行**之后**进行切分
- 例如：选择行号5，表示在第5行和第6行之间切分
- 优先选择段落结束、章节过渡、话题转换等自然边界
- 避免在句子中间或紧密相关的内容中间切分

```

请开始分析："""

def get_content_system_prompt() -> str:
    return """你是一个专业的文本内容整理专家。
你的任务是将给定文本拆分为多个独立完整的内容块，并为每个内容块生成问题。"""

def get_content_human_prompt(text: str) -> str:
    return f"""## 任务：文本内容拆分与问题生成

### 【待处理文本】
```txt
{text}
```

## 要求

### 内容块拆分规则
1. **完整性**：每个内容块必须是独立完整的，无需依赖其他内容或问题就能理解
2. **覆盖度**：所有内容块必须**完全覆盖**原始文本的全部信息，不能遗漏任何重要内容
3. **数量**：拆分为2-5个内容块（根据文本长度和复杂度决定）
4. **逻辑性**：按照原文的逻辑顺序拆分，保持叙述连贯
5. **独立性**：每个块应该是一个相对独立的知识点或话题

### 问题生成规则
1. **针对性**：问题必须基于对应内容块的核心内容
2. **可答性**：问题的答案应该在内容块中能找到

### 图像表述转换

当文本中包含图像描述时，需调整其表达，以确保内容块的完整性。例如原文为：

```txt
互联网交换点 IXP 的主要作用就是允许两个网络直接相连并交换分组，而不需要再通过第三个网络来转发分组。例如，在图 1-3 中右方的两个地区 ISP 通过一个 IXP 连接起来了。这样，主机 A 和主机 B 交换分组时，就不必再经过最上层的主干 ISP，而是直接在两个地区 ISP 之间用高速链路对等地交换分组。这样就使互联网上的数据流量分布更加合理，同时也减少了分组转发的迟延时间，降低了分组转发的费用。
```

由于图像并不能以文字表示，因此需要结合上下文调整这段内容的表达，调整为：

```txt
互联网交换点（IXP）的核心作用是允许两个或多个网络直接互联并交换数据分组，无需经过第三方网络转发。例如，两个地区级ISP可以通过IXP直接连接，使得各自网络中的主机（如主机A与主机B）之间的通信不必再绕经上层主干ISP，而是通过IXP内部的高速链路直接对等交换数据。这种直接互联模式能够优化互联网流量分布，减少分组转发延迟，并降低转发成本。
```

## 拆分示例

**原文**：
> 计算机网络是指将地理位置不同的具有独立功能的多台计算机通过通信线路连接起来，以实现资源共享和信息传递的系统。网络协议是计算机网络的核心，它规定了网络中数据交换的规则和标准。常见的网络协议有TCP/IP、HTTP、FTP等。

**内容块拆分**：
1. 内容块1：计算机网络是指将地理位置不同的具有独立功能的多台计算机通过通信线路连接起来，以实现资源共享和信息传递的系统。
   - 问题：什么是计算机网络？

2. 内容块2：网络协议是计算机网络的核心，它规定了网络中数据交换的规则和标准。
   - 问题：网络协议在计算机网络中的作用是什么？

3. 内容块3：常见的网络协议有TCP/IP、HTTP、FTP等。
   - 问题：常见的网络协议有哪些？

## 输出格式
```json
{{
  "content_blocks": [
    {{
      "content": "内容块1的完整文本",
      "question": "针对内容块1的问题？"
    }},
    {{
      "content": "内容块2的完整文本",
      "question": "针对内容块2的问题？"
    }},
    {{
      "content": "内容块3的完整文本",
      "question": "针对内容块3的问题？"
    }}
  ]
}}
```

请开始处理："""

# --- 3. 核心处理逻辑 ---
class BookProcessor:
    def __init__(self, 
                 provider: str = "openai",
                 model: str = "qwen-plus",
                 base_url: str = None,
                 api_key: str = None, 
                 temperature: float = 0):
        
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
        
        # 为两个任务分别创建结构化输出
        self.boundary_llm = self.llm.with_structured_output(BoundaryResult)
        self.content_llm = self.llm.with_structured_output(ContentBlockResult)
        
        # 初始化token统计变量
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # 使用langchain的递归文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=0,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
    
    def _prepare_numbered_lines(self, current_text: str, next_text: str) -> Tuple[List[str], List[str]]:
        """
        将边界文本按行分割并添加行号
        
        Returns:
            (带行号的行列表, 原始行列表)
        """
        # 合并两段文本，取中间部分作为边界区域
        boundary_text = current_text[-512:] + next_text[:512]
        
        # 按换行符分割，去掉空行
        lines = [line for line in boundary_text.split('\n') if line.strip()]
        
        # 添加行号
        numbered_lines = [f"[{i+1}] {line}" for i, line in enumerate(lines)]
        
        return numbered_lines, lines
    
    def _load_existing_dataset(self, output_file: str) -> Tuple[List[Dict], int]:
        """加载已有的数据集，返回数据集和下一个要处理的chunk索引"""
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            entries = data.get('entries', [])
            if entries:
                max_chunk_id = max(entry['chunk_id'] for entry in entries)
                next_chunk_idx = max_chunk_id + 1
                
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
    
    def _find_optimal_boundary(self, current_text: str, next_text: str) -> Tuple[int, float, str, Dict]:
        """
        任务1：使用LLM找出最佳文本边界（基于行号）
        
        Returns:
            (最佳行号, 评分, 该行内容, token统计信息)
        """
        numbered_lines, original_lines = self._prepare_numbered_lines(current_text, next_text)
        
        system_prompt = get_boundary_system_prompt()
        human_content = get_boundary_human_prompt(numbered_lines)
        
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
                response = self.boundary_llm.invoke(messages)
                
                token_stats["prompt_tokens"] = cb.prompt_tokens
                token_stats["completion_tokens"] = cb.completion_tokens
                token_stats["total_tokens"] = cb.total_tokens
                token_stats["total_cost"] = cb.total_cost
                
                self.total_prompt_tokens += cb.prompt_tokens
                self.total_completion_tokens += cb.completion_tokens
                self.total_tokens += cb.total_tokens
                self.total_cost += cb.total_cost
            
            result = response.model_dump()
            optimal_splits = result.get('optimal_splits', [])
            
            if optimal_splits:
                # 选择评分最高的边界
                best_split = max(optimal_splits, key=lambda x: x.get('score', 0))
                line_number = best_split.get('line_number', 0)
                score = best_split.get('score', 0)
                
                # 获取该行的内容
                line_content = original_lines[line_number - 1] if 0 < line_number <= len(original_lines) else ''
                
                return line_number, score, line_content, token_stats
            else:
                return 0, 0, '', token_stats
                
        except Exception as e:
            print(f"❌ 边界查找错误: {type(e).__name__}: {str(e)}")
            raise
    
    def _extract_content_blocks(self, text: str) -> Tuple[List[Dict], Dict]:
        """
        任务2：使用LLM提取内容块和生成问题
        
        Returns:
            (内容块列表, token统计信息)
        """
        system_prompt = get_content_system_prompt()
        human_content = get_content_human_prompt(text)
        
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
                response = self.content_llm.invoke(messages)
                
                token_stats["prompt_tokens"] = cb.prompt_tokens
                token_stats["completion_tokens"] = cb.completion_tokens
                token_stats["total_tokens"] = cb.total_tokens
                token_stats["total_cost"] = cb.total_cost
                
                self.total_prompt_tokens += cb.prompt_tokens
                self.total_completion_tokens += cb.completion_tokens
                self.total_tokens += cb.total_tokens
                self.total_cost += cb.total_cost
            
            result = response.model_dump()
            return result.get('content_blocks', []), token_stats
            
        except Exception as e:
            print(f"❌ 内容提取错误: {type(e).__name__}: {str(e)}")
            raise
    
    def _adjust_text_boundary_by_line(self, current_chunk: str, next_chunk: str, 
                                      boundary_line_number: int) -> Tuple[str, str]:
        """
        根据行号调整文本分割
        
        Returns:
            (调整后的当前文本, 调整后的下一段文本)
        """
        if boundary_line_number <= 0:
            return current_chunk, next_chunk
        
        # 获取边界区域的行
        boundary_text = current_chunk[-512:] + next_chunk[:512]
        lines = [line for line in boundary_text.split('\n') if line.strip()]
        
        if boundary_line_number > len(lines):
            return current_chunk, next_chunk
        
        # 找到切分行在原文中的位置
        boundary_line_content = lines[boundary_line_number - 1]
        
        # 在合并文本中找到该行的位置
        combined = current_chunk + "\n" + next_chunk
        
        # 尝试精确匹配该行内容
        if boundary_line_content in combined:
            # 找到该行结束的位置
            line_end_pos = combined.find(boundary_line_content) + len(boundary_line_content)
            
            # 查找该行后的第一个换行符
            next_newline = combined.find('\n', line_end_pos)
            if next_newline == -1:
                split_pos = len(combined)
            else:
                split_pos = next_newline
            
            adjusted_current = combined[:split_pos].strip()
            adjusted_next = combined[split_pos:].strip()
            
            return adjusted_current, adjusted_next
        
        # 如果找不到精确匹配，保持原样
        return current_chunk, next_chunk
    
    def process_book_to_dataset(self, file_path: str, output_file: str):
        """
        处理书籍文件生成训练数据集，分两步执行：
        1. 边界优化（基于行号）
        2. 内容拆分与问题生成
        """
        dataset, start_chunk_idx = self._load_existing_dataset(output_file)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        print(f"📖 文件加载完成，总长度: {len(full_text)} 字符")
        
        initial_chunks = self.text_splitter.split_text(full_text)
        print(f"✂️  初步分割完成，共 {len(initial_chunks)} 个块")
        
        chunk_idx = start_chunk_idx
        i = start_chunk_idx
        
        while i < len(initial_chunks):
            current_chunk = initial_chunks[i]
            
            if not current_chunk.strip():
                i += 1
                continue
            
            next_chunk = initial_chunks[i+1] if i < len(initial_chunks) - 1 else ""
            
            try:
                # 步骤1：边界优化（基于行号）
                print(f"\n🔍 [{i+1}/{len(initial_chunks)}] 步骤1: 查找最佳边界行号...")
                boundary_line, boundary_score, line_content, boundary_tokens = self._find_optimal_boundary(
                    current_chunk, next_chunk
                )
                print(f"   ✓ 边界行号: {boundary_line} | 评分: {boundary_score:.1f} | Tokens: {boundary_tokens['total_tokens']}")
                print(f"   📍 边界内容: {line_content[:50]}...")
                
                # 调整文本边界
                adjusted_current, adjusted_next = self._adjust_text_boundary_by_line(
                    current_chunk, next_chunk, boundary_line
                )
                
                # 如果下一块被修改了，更新chunks列表
                if i < len(initial_chunks) - 1 and adjusted_next != next_chunk:
                    initial_chunks[i+1] = adjusted_next
                
                boundary_adjusted = (adjusted_current != current_chunk)
                if boundary_adjusted:
                    print(f"   📏 边界已调整: {len(current_chunk)} → {len(adjusted_current)} 字符")
                
                # 步骤2：内容拆分与问题生成
                print(f"   ⚙️  步骤2: 提取内容块...")
                qa_pairs, content_tokens = self._extract_content_blocks(adjusted_current)
                print(f"   ✓ 提取 {len(qa_pairs)} 个QA对 | Tokens: {content_tokens['total_tokens']}")
                
                if not qa_pairs:
                    print(f"   ⚠️  未提取到有效QA，跳过")
                    i += 1
                    continue
                
                # 合并token统计
                total_tokens_used = {
                    "boundary_tokens": boundary_tokens,
                    "content_tokens": content_tokens,
                    "total_prompt_tokens": boundary_tokens['prompt_tokens'] + content_tokens['prompt_tokens'],
                    "total_completion_tokens": boundary_tokens['completion_tokens'] + content_tokens['completion_tokens'],
                    "total_tokens": boundary_tokens['total_tokens'] + content_tokens['total_tokens'],
                    "total_cost": boundary_tokens['total_cost'] + content_tokens['total_cost']
                }
                
                # 保存结果
                entry = {
                    "chunk_id": chunk_idx,
                    "source_text": adjusted_current,
                    "qa_pairs": qa_pairs,
                    "metadata": {
                        "chunk_length": len(adjusted_current),
                        "boundary_adjusted": boundary_adjusted,
                        "boundary_line_number": boundary_line,
                        "boundary_line_content": line_content,
                        "boundary_score": boundary_score,
                        "original_chunk": current_chunk,
                        "token_usage": total_tokens_used
                    }
                }
                dataset.append(entry)
                chunk_idx += 1
                
                # 立即保存
                self._save_dataset(dataset, output_file)
                
                progress = (i + 1) / len(initial_chunks) * 100
                print(f"   💾 已保存 | 进度: {progress:.1f}% | 累计成本: ${self.total_cost:.4f}")
                
            except Exception as e:
                print(f"   ❌ 处理出错: {e}")
            
            i += 1
        
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
    # 示例1: 使用OpenAI兼容接口 (如qwen-plus)，API key从环境变量加载
    processor_openai = BookProcessor(
        provider="openai",
        model="qwen-plus",
        temperature=0
    )
    
    # 示例2: 使用GLM (智谱AI)，API key从环境变量加载
    processor_glm = BookProcessor(
        provider="glm",
        model="glm-4-flash",
        temperature=0
    )
    
    # 示例3: 使用Gemini (Google AI)，API key从环境变量加载
    processor_gemini = BookProcessor(
        provider="gemini",
        model="gemini-2.5-flash",
        temperature=0
    )
    
    # 选择使用的处理器
    processor = processor_openai  # 或 processor_glm, processor_gemini
    
    pwd = Path(__file__).parent
    artifacts_dir = pwd.parent / "artifacts" / "dataset"
    # 处理书籍
    processor.process_book_to_dataset(
        file_path=pwd / "MinerU_markdown_计算机网络-第7版-谢希仁_20251223084758_2003266285337022464.md", 
        output_file=pwd / "book_split.jsonl"
    )