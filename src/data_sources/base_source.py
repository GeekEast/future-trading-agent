"""数据源基础抽象类"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from loguru import logger
import asyncio
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


class BaseDataSource(ABC):
    """数据源基础抽象类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logger.bind(source=name)
        self.client: Optional[httpx.AsyncClient] = None
        self.session_timeout = config.get('timeout', 30)
        self.max_retries = config.get('retry_count', 3)
        self.base_url = config.get('base_url', '')
        self.api_key = config.get('api_key', '')
        
    async def initialize(self) -> None:
        """初始化数据源"""
        self.logger.info(f"初始化数据源: {self.name}")
        
        # 创建HTTP客户端
        self.client = httpx.AsyncClient(
            timeout=self.session_timeout,
            headers=self._get_default_headers()
        )
        
        # 执行自定义初始化
        await self._custom_initialize()
        
    async def cleanup(self) -> None:
        """清理资源"""
        if self.client:
            await self.client.aclose()
            self.client = None
        self.logger.info(f"数据源 {self.name} 清理完成")
        
    def _get_default_headers(self) -> Dict[str, str]:
        """获取默认HTTP头"""
        headers = {
            'User-Agent': 'Future-Trading-Agent/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        if self.api_key:
            headers.update(self._get_auth_headers())
            
        return headers
        
    def _get_auth_headers(self) -> Dict[str, str]:
        """获取认证头，子类可重写"""
        return {'Authorization': f'Bearer {self.api_key}'}
        
    async def _custom_initialize(self) -> None:
        """自定义初始化逻辑，子类可重写"""
        pass
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_request(
        self, 
        method: str, 
        url: str, 
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """发送HTTP请求"""
        if not self.client:
            raise RuntimeError(f"数据源 {self.name} 未初始化")
            
        try:
            response = await self.client.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP错误 {e.response.status_code}: {e.response.text}")
            raise
        except httpx.RequestError as e:
            self.logger.error(f"请求错误: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"未知错误: {str(e)}")
            raise
            
    @abstractmethod
    async def fetch_data(
        self, 
        symbols: List[str], 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """获取数据的抽象方法"""
        pass
        
    @abstractmethod
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """验证数据质量"""
        pass
        
    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """获取支持的标的列表"""
        pass
        
    @abstractmethod
    def get_rate_limit(self) -> Dict[str, int]:
        """获取速率限制信息"""
        pass
        
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 子类可重写此方法实现具体的健康检查逻辑
            return await self._perform_health_check()
        except Exception as e:
            self.logger.error(f"健康检查失败: {str(e)}")
            return False
            
    async def _perform_health_check(self) -> bool:
        """执行健康检查的默认实现"""
        return True
        
    def get_metadata(self) -> Dict[str, Any]:
        """获取数据源元数据"""
        return {
            'name': self.name,
            'base_url': self.base_url,
            'timeout': self.session_timeout,
            'max_retries': self.max_retries,
            'supported_symbols': self.get_supported_symbols(),
            'rate_limit': self.get_rate_limit()
        }
        
    def __str__(self) -> str:
        return f"DataSource({self.name})"
        
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>" 