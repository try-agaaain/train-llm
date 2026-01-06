"""
环境变量加载工具
使用pathlib从相对路径加载.env文件
"""

import os
from pathlib import Path
from dotenv import load_dotenv


def load_env(current_file: str) -> bool:
    """
    从项目根目录加载.env文件
    
    Args:
        current_file: 调用文件的__file__路径
        
    Returns:
        bool: 是否成功加载.env文件
    """
    # 获取调用文件的路径
    current_path = Path(current_file).resolve()
    
    # 向上查找到项目根目录（包含.env文件的目录）
    # 假设项目结构为 minimind-chat/.env
    project_root = current_path
    while project_root.parent != project_root:
        env_file = project_root / ".env"
        if env_file.exists():
            # 加载.env文件
            success = load_dotenv(env_file)
            if success:
                print(f"✅ 成功加载环境变量: {env_file}")
            return success
        project_root = project_root.parent
    
    print("⚠️ 未找到.env文件")
    return False


def get_api_key(provider: str = "QWEN") -> str:
    """
    获取指定服务商的API key
    
    Args:
        provider: 服务商名称，如 "QWEN", "GEMINI", "GLM"
        
    Returns:
        str: API key，如果未找到则返回空字符串
    """
    key_name = f"{provider.upper()}_API_KEY"
    api_key = os.getenv(key_name, "")
    
    if not api_key:
        print(f"⚠️ 未找到环境变量: {key_name}")
    
    return api_key


def get_base_url(provider: str = "QWEN") -> str:
    """
    获取指定服务商的base URL
    
    Args:
        provider: 服务商名称，如 "QWEN", "GEMINI", "GLM"
        
    Returns:
        str: base URL，如果未找到则返回默认值
    """
    # 默认base URL映射
    default_urls = {
        "QWEN": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "GLM": "https://open.bigmodel.cn/api/paas/v4",
        "GEMINI": "https://generativelanguage.googleapis.com",
        "LOCAL": "http://localhost:8000/v1"
    }
    
    # 先尝试从环境变量获取
    key_name = f"{provider.upper()}_BASE_URL"
    base_url = os.getenv(key_name, "")
    
    # 如果环境变量不存在，使用默认值
    if not base_url:
        base_url = default_urls.get(provider.upper(), "")
    
    return base_url


def get_artifacts_dir(custom_path: str = None, current_file: str = None) -> Path:
    """
    获取artifacts目录路径，优先级：参数传递 > 环境变量 > 默认值
    
    Args:
        custom_path: 自定义路径（通常来自命令行参数）
        current_file: 调用文件的__file__路径（用于计算默认路径）
        
    Returns:
        Path: artifacts目录的绝对路径
    """
    # 优先级1: 命令行参数
    if custom_path:
        return Path(custom_path).resolve()
    
    # 优先级2: 环境变量
    env_path = os.getenv("ARTIFACTS_DIR", "")
    if env_path:
        return Path(env_path).resolve()
    
    # 优先级3: 默认值（项目根目录/artifacts）
    if current_file:
        # 从当前文件向上查找项目根目录
        current_path = Path(current_file).resolve()
        project_root = current_path
        while project_root.parent != project_root:
            # 检查是否是项目根目录（包含pyproject.toml或.env）
            if (project_root / "pyproject.toml").exists() or (project_root / ".env").exists():
                return project_root / "artifacts"
            project_root = project_root.parent
    
    # 如果没有找到项目根目录，使用当前工作目录
    return Path.cwd() / "artifacts"
