from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_env, get_api_key, get_base_url

# 加载环境变量
load_env(__file__)

model = init_chat_model(
    model="minimind",
    base_url=get_base_url("LOCAL"),
    api_key=get_api_key("LOCAL"),
    model_provider="openai"
)

system_prompt = f"""
一起聊天吧
"""

agent = create_agent(
    model=model,
    tools=[DuckDuckGoSearchRun(name="web search", description="当你需要获取最新的或者具体的信息时，使用这个工具进行网络搜索")],
    system_prompt=system_prompt,

)
