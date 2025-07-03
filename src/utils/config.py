"""配置管理模块"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    # 加载环境变量文件
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 替换环境变量
    config = _replace_env_vars(config)
    
    return config


def _replace_env_vars(data: Any) -> Any:
    """递归替换配置中的环境变量"""
    if isinstance(data, dict):
        return {k: _replace_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_replace_env_vars(item) for item in data]
    elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
        env_var = data[2:-1]
        return os.getenv(env_var, data)
    return data
