"""FRED API数据源 - 美联储经济数据"""

from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
from fredapi import Fred

from .base_source import BaseDataSource
from models.macro_events import EconomicIndicator, EconomicIndicatorType, EventImportance
from utils.helpers import convert_to_numeric, ValidationError


class FredDataSource(BaseDataSource):
    """FRED (Federal Reserve Economic Data) 数据源"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("FRED", config)
        self.fred_client: Optional[Fred] = None
        
        # FRED数据系列映射
        self.data_series = {
            # 利率相关
            'FEDFUNDS': {'name': '联邦基金利率', 'type': EconomicIndicatorType.MONETARY},
            'DFF': {'name': '有效联邦基金利率', 'type': EconomicIndicatorType.MONETARY},
            'DGS10': {'name': '10年期国债收益率', 'type': EconomicIndicatorType.MONETARY},
            'DGS2': {'name': '2年期国债收益率', 'type': EconomicIndicatorType.MONETARY},
            'DGS1MO': {'name': '1个月期国债收益率', 'type': EconomicIndicatorType.MONETARY},
            
            # 就业数据
            'UNRATE': {'name': '失业率', 'type': EconomicIndicatorType.EMPLOYMENT},
            'PAYEMS': {'name': '非农就业人数', 'type': EconomicIndicatorType.EMPLOYMENT},
            'ICSA': {'name': '初请失业金人数', 'type': EconomicIndicatorType.EMPLOYMENT},
            'CIVPART': {'name': '劳动参与率', 'type': EconomicIndicatorType.EMPLOYMENT},
            
            # 通胀数据
            'CPIAUCSL': {'name': 'CPI消费者价格指数', 'type': EconomicIndicatorType.INFLATION},
            'CPILFESL': {'name': '核心CPI', 'type': EconomicIndicatorType.INFLATION},
            'PCEPI': {'name': 'PCE物价指数', 'type': EconomicIndicatorType.INFLATION},
            'PCEPILFE': {'name': '核心PCE', 'type': EconomicIndicatorType.INFLATION},
            
            # 经济增长
            'GDP': {'name': '国内生产总值', 'type': EconomicIndicatorType.GROWTH},
            'GDPC1': {'name': '实际GDP', 'type': EconomicIndicatorType.GROWTH},
            'INDPRO': {'name': '工业生产指数', 'type': EconomicIndicatorType.GROWTH},
            'RSAFS': {'name': '零售销售', 'type': EconomicIndicatorType.CONSUMER},
            
            # 货币政策
            'WALCL': {'name': '美联储资产负债表', 'type': EconomicIndicatorType.MONETARY},
            'WRESBAL': {'name': '银行准备金', 'type': EconomicIndicatorType.MONETARY},
            'RRPONTSYD': {'name': '隔夜逆回购', 'type': EconomicIndicatorType.MONETARY},
            
            # 制造业
            'NAPM': {'name': 'ISM制造业PMI', 'type': EconomicIndicatorType.MANUFACTURING},
            'NAPMNOI': {'name': 'ISM制造业新订单', 'type': EconomicIndicatorType.MANUFACTURING},
            
            # 房地产
            'HOUST': {'name': '新屋开工', 'type': EconomicIndicatorType.HOUSING},
            'HSNGMI': {'name': '新屋销售', 'type': EconomicIndicatorType.HOUSING},
            
            # 消费者信心
            'UMCSENT': {'name': '密歇根消费者信心指数', 'type': EconomicIndicatorType.SENTIMENT},
        }
        
    async def _custom_initialize(self) -> None:
        """初始化FRED客户端"""
        try:
            api_key = self.config.get('api_key')
            if not api_key:
                raise ValidationError("FRED API密钥未配置")
                
            self.fred_client = Fred(api_key=api_key)
            self.logger.info("FRED客户端初始化成功")
            
        except Exception as e:
            self.logger.error(f"FRED客户端初始化失败: {e}")
            raise
            
    async def fetch_data(
        self, 
        symbols: List[str], 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """获取FRED数据"""
        if not self.fred_client:
            raise RuntimeError("FRED客户端未初始化")
            
        # 设置默认日期范围
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)  # 默认1年数据
            
        results = {
            'source': 'FRED',
            'timestamp': datetime.now(),
            'data': [],
            'metadata': {
                'start_date': start_date,
                'end_date': end_date,
                'symbols_requested': symbols,
                'symbols_available': list(self.data_series.keys())
            }
        }
        
        for symbol in symbols:
            try:
                if symbol not in self.data_series:
                    self.logger.warning(f"未知的FRED系列: {symbol}")
                    continue
                    
                # 获取数据
                data = await self._fetch_series_data(symbol, start_date, end_date)
                if data:
                    results['data'].extend(data)
                    
            except Exception as e:
                self.logger.error(f"获取FRED数据失败 {symbol}: {e}")
                continue
                
        return results
        
    async def _fetch_series_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[EconomicIndicator]:
        """获取单个数据系列"""
        try:
            series_info = self.data_series[symbol]
            
            # 获取数据
            data = self.fred_client.get_series(
                symbol, 
                start=start_date.date(), 
                end=end_date.date()
            )
            
            if data.empty:
                self.logger.warning(f"FRED系列 {symbol} 无数据")
                return []
                
            # 获取系列信息
            series_meta = self.fred_client.get_series_info(symbol)
            
            indicators = []
            previous_value = None
            
            for date_idx, value in data.items():
                if pd.isna(value):
                    continue
                    
                # 创建经济指标对象
                indicator = EconomicIndicator(
                    name=series_info['name'],
                    symbol=symbol,
                    value=float(value),
                    unit=series_meta.get('units', ''),
                    frequency=series_meta.get('frequency', ''),
                    release_date=datetime.combine(date_idx, datetime.min.time()),
                    previous_value=previous_value,
                    forecast_value=None,  # FRED不提供预测值
                    revision=None,
                    importance=self._determine_importance(symbol),
                    indicator_type=series_info['type']
                )
                
                indicators.append(indicator)
                previous_value = float(value)
                
            self.logger.info(f"获取FRED数据 {symbol}: {len(indicators)}条记录")
            return indicators
            
        except Exception as e:
            self.logger.error(f"获取FRED系列数据失败 {symbol}: {e}")
            return []
            
    def _determine_importance(self, symbol: str) -> EventImportance:
        """根据指标类型确定重要性"""
        high_importance = [
            'FEDFUNDS', 'DFF', 'UNRATE', 'PAYEMS', 'CPIAUCSL', 'CPILFESL',
            'PCEPI', 'GDP', 'GDPC1', 'NAPM'
        ]
        
        medium_importance = [
            'DGS10', 'DGS2', 'ICSA', 'INDPRO', 'RSAFS', 'WALCL', 'UMCSENT'
        ]
        
        if symbol in high_importance:
            return EventImportance.HIGH
        elif symbol in medium_importance:
            return EventImportance.MEDIUM
        else:
            return EventImportance.LOW
            
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """验证数据质量"""
        try:
            if not data.get('data'):
                return False
                
            # 检查数据结构
            for item in data['data']:
                if not isinstance(item, EconomicIndicator):
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"数据验证失败: {e}")
            return False
            
    def get_supported_symbols(self) -> List[str]:
        """获取支持的标的列表"""
        return list(self.data_series.keys())
        
    def get_rate_limit(self) -> Dict[str, int]:
        """获取速率限制信息"""
        return {
            'requests_per_day': 120000,  # FRED API限制
            'requests_per_minute': 120,
            'concurrent_requests': 10
        }
        
    async def _perform_health_check(self) -> bool:
        """执行健康检查"""
        try:
            if not self.fred_client:
                return False
                
            # 测试获取一个简单的数据系列
            test_data = self.fred_client.get_series('FEDFUNDS', limit=1)
            return not test_data.empty
            
        except Exception as e:
            self.logger.error(f"FRED健康检查失败: {e}")
            return False
            
    async def get_latest_data(self, symbols: List[str]) -> Dict[str, Any]:
        """获取最新数据"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 最近30天
        
        return await self.fetch_data(symbols, start_date, end_date)
        
    async def get_key_indicators(self) -> Dict[str, Any]:
        """获取关键经济指标"""
        key_symbols = [
            'FEDFUNDS',  # 联邦基金利率
            'UNRATE',    # 失业率
            'CPIAUCSL',  # CPI
            'GDP',       # GDP
            'DGS10',     # 10年期国债
            'NAPM'       # ISM PMI
        ]
        
        return await self.get_latest_data(key_symbols)
        
    def get_data_series_info(self) -> Dict[str, Dict[str, str]]:
        """获取数据系列信息"""
        return self.data_series 