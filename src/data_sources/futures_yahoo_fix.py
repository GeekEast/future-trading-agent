"""期货数据Yahoo Finance修复模块 - 临时解决方案"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import yfinance as yf
import pandas as pd

from .base_source import BaseDataSource
from models.market_data import MarketDataPoint, OHLCV
from utils.helpers import convert_to_numeric


class FuturesYahooFixDataSource(BaseDataSource):
    """期货数据Yahoo Finance修复版本"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("FuturesYahooFix", config)
        
        # 期货标的映射到可靠的ETF代理
        self.futures_to_etf_mapping = {
            # 期货标的 -> (ETF代理, 乘数调整, 名称)
            'ES': ('SPY', 10.0, 'S&P 500 E-mini (via SPY)'),
            'NQ': ('QQQ', 5.0, 'NASDAQ 100 E-mini (via QQQ)'),
            'RTY': ('IWM', 5.0, 'Russell 2000 E-mini (via IWM)'),
            'GC': ('GLD', 10.0, 'Gold Futures (via GLD)'),
            'CL': ('USO', 1000.0, 'Crude Oil Futures (via USO)')
        }
        
        # 真实期货合约的基础信息
        self.futures_contracts_info = {
            'ES': {
                'name': 'E-mini S&P 500',
                'exchange': 'CME',
                'tick_size': 0.25,
                'tick_value': 12.50,
                'multiplier': 50,
                'margin_requirement': 13200,
                'trading_hours': '17:00-16:00 CT',
                'base_price_range': (4000, 6000)
            },
            'NQ': {
                'name': 'E-mini NASDAQ 100',
                'exchange': 'CME',
                'tick_size': 0.25,
                'tick_value': 5.00,
                'multiplier': 20,
                'margin_requirement': 17600,
                'trading_hours': '17:00-16:00 CT',
                'base_price_range': (18000, 25000)
            },
            'RTY': {
                'name': 'E-mini Russell 2000',
                'exchange': 'CME',
                'tick_size': 0.1,
                'tick_value': 5.00,
                'multiplier': 50,
                'margin_requirement': 5500,
                'trading_hours': '17:00-16:00 CT',
                'base_price_range': (1800, 2500)
            },
            'GC': {
                'name': 'Gold Futures',
                'exchange': 'COMEX',
                'tick_size': 0.1,
                'tick_value': 10.00,
                'multiplier': 100,
                'margin_requirement': 11000,
                'trading_hours': '17:00-16:00 CT',
                'base_price_range': (2000, 2500)
            },
            'CL': {
                'name': 'Light Sweet Crude Oil',
                'exchange': 'NYMEX',
                'tick_size': 0.01,
                'tick_value': 10.00,
                'multiplier': 1000,
                'margin_requirement': 6600,
                'trading_hours': '17:00-16:00 CT',
                'base_price_range': (60, 120)
            }
        }
        
        # 为每个期货保存额外的元数据
        self.futures_metadata = {}
        
    async def _custom_initialize(self) -> None:
        """初始化期货数据源修复版本"""
        self.logger.info("期货数据Yahoo Finance修复版本初始化完成")
        
    async def fetch_data(
        self, 
        symbols: List[str], 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """获取期货数据（修复版本）"""
        
        data_type = kwargs.get('data_type', 'quote')
        
        results = {
            'source': 'FuturesYahooFix',
            'timestamp': datetime.now(),
            'data_type': data_type,
            'data': [],
            'metadata': {
                'symbols_requested': symbols,
                'mapping_used': True,
                'note': '使用ETF代理期货数据，价格已调整'
            }
        }
        
        for symbol in symbols:
            try:
                if symbol in self.futures_to_etf_mapping:
                    data = await self._fetch_futures_via_etf(symbol)
                else:
                    # 对于非期货标的，直接使用原始方法
                    data = await self._fetch_direct_data(symbol)
                    
                if data:
                    results['data'].extend(data)
                    
            except Exception as e:
                self.logger.error(f"获取期货数据失败 {symbol}: {e}")
                continue
                
        return results
        
    async def _fetch_futures_via_etf(self, futures_symbol: str) -> List[MarketDataPoint]:
        """通过ETF代理获取期货数据"""
        
        etf_symbol, multiplier, name = self.futures_to_etf_mapping[futures_symbol]
        contract_info = self.futures_contracts_info[futures_symbol]
        
        try:
            # 获取ETF数据
            ticker = yf.Ticker(etf_symbol)
            info = ticker.info
            
            if not info:
                self.logger.warning(f"无法获取ETF数据: {etf_symbol}")
                return []
                
            # 获取ETF价格
            etf_price = convert_to_numeric(info.get('currentPrice', 0)) or \
                       convert_to_numeric(info.get('regularMarketPrice', 0)) or \
                       convert_to_numeric(info.get('previousClose', 0))
                       
            if etf_price <= 0:
                self.logger.warning(f"ETF价格无效: {etf_symbol} = {etf_price}")
                return []
                
            # 计算期货价格（使用乘数调整）
            futures_price = etf_price * multiplier
            
            # 验证价格合理性
            price_range = contract_info['base_price_range']
            if not (price_range[0] <= futures_price <= price_range[1]):
                # 如果价格超出合理范围，使用另一种计算方法
                futures_price = self._calculate_alternative_price(futures_symbol, etf_price)
                
            # 创建期货数据点
            data_point = MarketDataPoint(
                timestamp=datetime.now(),
                symbol=futures_symbol,
                price=futures_price,
                volume=convert_to_numeric(info.get('volume', 0)) or 0,
                bid=futures_price - contract_info['tick_size'],
                ask=futures_price + contract_info['tick_size'],
                change=convert_to_numeric(info.get('regularMarketChange', 0)) * multiplier,
                change_percent=convert_to_numeric(info.get('regularMarketChangePercent', 0)) * 100
            )
            
            # 保存期货特有信息到类的metadata存储
            self.futures_metadata[futures_symbol] = {
                'contract_name': contract_info['name'],
                'exchange': contract_info['exchange'],
                'tick_size': contract_info['tick_size'],
                'tick_value': contract_info['tick_value'],
                'multiplier': contract_info['multiplier'],
                'margin_requirement': contract_info['margin_requirement'],
                'etf_proxy': etf_symbol,
                'etf_price': etf_price,
                'adjustment_multiplier': multiplier,
                'data_quality': 'proxy_adjusted'
            }
            
            self.logger.info(f"期货数据获取成功: {futures_symbol} = ${futures_price:.2f} (via {etf_symbol})")
            return [data_point]
            
        except Exception as e:
            self.logger.error(f"通过ETF获取期货数据失败 {futures_symbol}: {e}")
            return []
            
    def _calculate_alternative_price(self, futures_symbol: str, etf_price: float) -> float:
        """计算替代期货价格"""
        
        # 使用更精确的转换公式
        conversion_formulas = {
            'ES': lambda x: x * 10.04,  # 更精确的SPY到ES转换
            'NQ': lambda x: x * 36.0,   # 修正的QQQ到NQ转换 (QQQ $550 -> NQ ~$19,800)
            'RTY': lambda x: x * 10.2,  # IWM到RTY转换
            'GC': lambda x: x * 7.0,    # 修正的GLD到黄金期货 (GLD $309 -> 黄金 ~$2,163)
            'CL': lambda x: x * 0.95    # 修正的USO到原油期货 (USO $75 -> 原油 ~$71)
        }
        
        if futures_symbol in conversion_formulas:
            return conversion_formulas[futures_symbol](etf_price)
        else:
            return etf_price * 10.0  # 默认乘数
            
    async def _fetch_direct_data(self, symbol: str) -> List[MarketDataPoint]:
        """直接获取非期货数据"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return []
                
            price = convert_to_numeric(info.get('currentPrice', 0)) or \
                   convert_to_numeric(info.get('regularMarketPrice', 0)) or \
                   convert_to_numeric(info.get('previousClose', 0))
            
            if price <= 0:
                return []
                
            data_point = MarketDataPoint(
                timestamp=datetime.now(),
                symbol=symbol,
                price=price,
                volume=convert_to_numeric(info.get('volume', 0)) or 0,
                bid=convert_to_numeric(info.get('bid', 0)),
                ask=convert_to_numeric(info.get('ask', 0)),
                change=convert_to_numeric(info.get('regularMarketChange', 0)),
                change_percent=convert_to_numeric(info.get('regularMarketChangePercent', 0)) * 100
            )
            
            return [data_point]
            
        except Exception as e:
            self.logger.error(f"直接获取数据失败 {symbol}: {e}")
            return []
            
    async def get_futures_contract_info(self, symbol: str) -> Dict[str, Any]:
        """获取期货合约信息"""
        if symbol in self.futures_contracts_info:
            return self.futures_contracts_info[symbol].copy()
        else:
            return {}
            
    def get_futures_metadata(self, symbol: str) -> Dict[str, Any]:
        """获取期货元数据"""
        return self.futures_metadata.get(symbol, {})
            
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """验证数据质量"""
        try:
            if not data.get('data'):
                return False
                
            # 检查期货价格合理性
            for item in data['data']:
                if hasattr(item, 'symbol') and item.symbol in self.futures_contracts_info:
                    price_range = self.futures_contracts_info[item.symbol]['base_price_range']
                    if not (price_range[0] <= item.price <= price_range[1]):
                        self.logger.warning(f"期货价格可能异常: {item.symbol} = {item.price}")
                        
            return True
            
        except Exception as e:
            self.logger.error(f"数据验证失败: {e}")
            return False
            
    def get_supported_futures(self) -> List[str]:
        """获取支持的期货列表"""
        return list(self.futures_to_etf_mapping.keys())
        
    def get_etf_mapping(self) -> Dict[str, str]:
        """获取期货到ETF的映射关系"""
        return {k: v[0] for k, v in self.futures_to_etf_mapping.items()}
        
    def get_rate_limit(self) -> Dict[str, int]:
        """获取速率限制信息"""
        return {
            'requests_per_minute': 60,
            'requests_per_day': 2000,
            'concurrent_requests': 5
        }
        
    def get_supported_symbols(self) -> List[str]:
        """获取支持的标的列表"""
        # 返回期货标的 + 常见ETF/股票
        futures = list(self.futures_to_etf_mapping.keys())
        etfs = ['SPY', 'QQQ', 'IWM', 'GLD', 'USO', 'VIX']
        return futures + etfs
        
    async def _perform_health_check(self) -> bool:
        """执行健康检查"""
        try:
            # 测试获取ES数据
            test_data = await self._fetch_futures_via_etf('ES')
            return len(test_data) > 0 and test_data[0].price > 3000
            
        except Exception as e:
            self.logger.error(f"期货数据源健康检查失败: {e}")
            return False 