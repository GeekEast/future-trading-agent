#!/usr/bin/env python3
"""
期货交易AI Agent主程序
基于LangChain+LangGraph的盘前分析系统
"""

import asyncio
import click
from loguru import logger
from utils.config import load_config
from utils.logger import setup_logger


@click.command()
@click.option('--config', '-c', default='config/settings.yaml', help='配置文件路径')
@click.option('--debug', '-d', is_flag=True, help='调试模式')
@click.option('--agent', '-a', type=click.Choice(['macro', 'sentiment', 'option', 'all']), 
              default='all', help='运行指定的Agent')
def main(config, debug, agent):
    """期货交易AI Agent主程序"""
    
    # 加载配置
    settings = load_config(config)
    
    # 设置日志
    setup_logger(debug=debug)
    
    logger.info("🚀 启动期货交易AI Agent...")
    logger.info(f"配置文件: {config}")
    logger.info(f"调试模式: {debug}")
    logger.info(f"运行Agent: {agent}")
    
    # TODO: 实现Agent运行逻辑
    logger.info("✅ Agent运行完成")


if __name__ == "__main__":
    main()
