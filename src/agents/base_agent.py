"""Agent基础抽象类"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from loguru import logger


class BaseAgent(ABC):
    """Agent基础抽象类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logger.bind(agent=name)
        
    @abstractmethod
    async def initialize(self) -> None:
        """初始化Agent"""
        pass
        
    @abstractmethod
    async def fetch_data(self) -> Dict[str, Any]:
        """获取数据"""
        pass
        
    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析数据"""
        pass
        
    @abstractmethod
    async def generate_report(self, analysis: Dict[str, Any]) -> str:
        """生成报告"""
        pass
        
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """获取依赖的其他Agent"""
        pass
        
    async def run(self) -> Dict[str, Any]:
        """运行Agent"""
        self.logger.info(f"🚀 启动Agent: {self.name}")
        
        try:
            # 初始化
            await self.initialize()
            
            # 获取数据
            data = await self.fetch_data()
            
            # 分析数据
            analysis = await self.analyze(data)
            
            # 生成报告
            report = await self.generate_report(analysis)
            
            self.logger.info(f"✅ Agent {self.name} 运行完成")
            
            return {
                "agent": self.name,
                "data": data,
                "analysis": analysis,
                "report": report
            }
            
        except Exception as e:
            self.logger.error(f"❌ Agent {self.name} 运行失败: {str(e)}")
            raise
