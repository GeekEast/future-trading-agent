"""Alpha Vantage API数据源 - 股票和市场数据"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import asyncio
import json

from .base_source import BaseDataSource
from models.market_data import (
    MarketDataPoint, OHLCV, MarketIndex, AssetType, 
    MarketDataType, MarketDataCollection
)
from utils.helpers import convert_to_numeric, ValidationError, async_retry


class AlphaVantageDataSource(BaseDataSource):
    """Alpha Vantage API 数据源"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("AlphaVantage", config)
        self.base_url = "https://www.alphavantage.co/query"
        
        # 支持的功能映射
        self.functions = {
            'quote': 'GLOBAL_QUOTE',
            'intraday': 'TIME_SERIES_INTRADAY', 
            'daily': 'TIME_SERIES_DAILY',
            'weekly': 'TIME_SERIES_WEEKLY',
            'monthly': 'TIME_SERIES_MONTHLY',
            'sma': 'SMA',  # 简单移动平均
            'ema': 'EMA',  # 指数移动平均
            'rsi': 'RSI',  # 相对强弱指数
            'macd': 'MACD', # MACD指标
            'bbands': 'BBANDS', # 布林带
            'news': 'NEWS_SENTIMENT'
        }
        
        # 支持的股票和ETF
        self.supported_symbols = [
            # 主要指数ETF
            'SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO',
            # 期货相关ETF
            'GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBB',
            # 波动率相关
            'VIX', 'UVXY', 'SVXY',
            # 美股主要股票
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA',
            # 金融股
            'JPM', 'BAC', 'WFC', 'GS', 'MS'
        ]
        
    def _get_auth_headers(self) -> Dict[str, str]:
        """Alpha Vantage使用URL参数认证，不需要特殊头"""
        return {}
        
    async def fetch_data(
        self, 
        symbols: List[str], 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """获取市场数据"""
        
        data_type = kwargs.get('data_type', 'quote')
        interval = kwargs.get('interval', '1min')
        
        results = {
            'source': 'AlphaVantage',
            'timestamp': datetime.now(),
            'data_type': data_type,
            'data': [],
            'metadata': {
                'symbols_requested': symbols,
                'function': self.functions.get(data_type, 'GLOBAL_QUOTE'),
                'interval': interval
            }
        }
        
        # 控制请求频率 (Alpha Vantage有严格的频率限制)
        delay_between_requests = 12  # 每分钟5次请求
        
        for i, symbol in enumerate(symbols):
            try:
                if i > 0:
                    await asyncio.sleep(delay_between_requests)
                    
                data = await self._fetch_symbol_data(symbol, data_type, **{k: v for k, v in kwargs.items() if k != 'data_type'})
                if data:
                    results['data'].extend(data)
                    
            except Exception as e:
                self.logger.error(f"获取Alpha Vantage数据失败 {symbol}: {e}")
                continue
                
        return results
        
    @async_retry(max_retries=3, delay=1.0)
    async def _fetch_symbol_data(
        self, 
        symbol: str, 
        data_type: str, 
        **kwargs
    ) -> List[Any]:
        """获取单个标的数据"""
        
        params = {
            'function': self.functions.get(data_type, 'GLOBAL_QUOTE'),
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        # 根据数据类型添加参数
        if data_type == 'intraday':
            params['interval'] = kwargs.get('interval', '1min')
            params['outputsize'] = 'compact'
        elif data_type in ['sma', 'ema', 'rsi']:
            params['interval'] = kwargs.get('interval', 'daily')
            params['time_period'] = kwargs.get('time_period', 20)
            params['series_type'] = kwargs.get('series_type', 'close')
        elif data_type == 'macd':
            params['interval'] = kwargs.get('interval', 'daily')
            params['series_type'] = 'close'
        elif data_type == 'bbands':
            params['interval'] = kwargs.get('interval', 'daily')
            params['time_period'] = kwargs.get('time_period', 20)
            params['series_type'] = 'close'
            
        try:
            response = await self._make_request('GET', self.base_url, params=params)
            return await self._parse_response(symbol, data_type, response)
            
        except Exception as e:
            self.logger.error(f"请求Alpha Vantage失败 {symbol}: {e}")
            raise
            
    async def _parse_response(
        self, 
        symbol: str, 
        data_type: str, 
        response: Dict[str, Any]
    ) -> List[Any]:
        """解析API响应"""
        
        if 'Error Message' in response:
            raise ValueError(f"API错误: {response['Error Message']}")
            
        if 'Note' in response:
            raise ValueError(f"API限制: {response['Note']}")
            
        try:
            if data_type == 'quote':
                return await self._parse_quote_data(symbol, response)
            elif data_type in ['daily', 'weekly', 'monthly', 'intraday']:
                return await self._parse_time_series_data(symbol, response, data_type)
            elif data_type in ['sma', 'ema', 'rsi', 'macd', 'bbands']:
                return await self._parse_technical_data(symbol, response, data_type)
            else:
                self.logger.warning(f"未知数据类型: {data_type}")
                return []
                
        except Exception as e:
            self.logger.error(f"解析响应失败 {symbol}: {e}")
            return []
            
    async def _parse_quote_data(
        self, 
        symbol: str, 
        response: Dict[str, Any]
    ) -> List[MarketDataPoint]:
        """解析实时报价数据"""
        
        quote_key = 'Global Quote'
        if quote_key not in response:
            return []
            
        quote = response[quote_key]
        
        try:
            data_point = MarketDataPoint(
                timestamp=datetime.now(),
                symbol=symbol,
                price=convert_to_numeric(quote.get('05. price', 0)),
                volume=convert_to_numeric(quote.get('06. volume', 0)),
                change=convert_to_numeric(quote.get('09. change', 0)),
                change_percent=convert_to_numeric(
                    quote.get('10. change percent', '0%').replace('%', '')
                )
            )
            
            return [data_point]
            
        except Exception as e:
            self.logger.error(f"解析报价数据失败: {e}")
            return []
            
    async def _parse_time_series_data(
        self, 
        symbol: str, 
        response: Dict[str, Any], 
        data_type: str
    ) -> List[OHLCV]:
        """解析时间序列数据"""
        
        # 找到时间序列数据键
        time_series_key = None
        for key in response.keys():
            if 'Time Series' in key:
                time_series_key = key
                break
                
        if not time_series_key or time_series_key not in response:
            return []
            
        time_series = response[time_series_key]
        ohlcv_data = []
        
        for date_str, values in time_series.items():
            try:
                timestamp = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') \
                    if ' ' in date_str else datetime.strptime(date_str, '%Y-%m-%d')
                
                ohlcv = OHLCV(
                    timestamp=timestamp,
                    open=convert_to_numeric(values.get('1. open', 0)),
                    high=convert_to_numeric(values.get('2. high', 0)),
                    low=convert_to_numeric(values.get('3. low', 0)),
                    close=convert_to_numeric(values.get('4. close', 0)),
                    volume=convert_to_numeric(values.get('5. volume', 0))
                )
                
                ohlcv_data.append(ohlcv)
                
            except Exception as e:
                self.logger.warning(f"解析时间序列数据点失败 {date_str}: {e}")
                continue
                
        return sorted(ohlcv_data, key=lambda x: x.timestamp)
        
    async def _parse_technical_data(
        self, 
        symbol: str, 
        response: Dict[str, Any], 
        indicator: str
    ) -> List[Dict[str, Any]]:
        """解析技术指标数据"""
        
        # 找到技术指标数据键
        technical_key = None
        for key in response.keys():
            if 'Technical Analysis' in key or indicator.upper() in key:
                technical_key = key
                break
                
        if not technical_key or technical_key not in response:
            return []
            
        technical_data = response[technical_key]
        results = []
        
        for date_str, values in technical_data.items():
            try:
                timestamp = datetime.strptime(date_str, '%Y-%m-%d')
                
                result = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'indicator': indicator,
                    'values': {}
                }
                
                # 解析不同指标的值
                for key, value in values.items():
                    clean_key = key.split('. ')[-1] if '. ' in key else key
                    result['values'][clean_key] = convert_to_numeric(value)
                    
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"解析技术指标数据失败 {date_str}: {e}")
                continue
                
        return sorted(results, key=lambda x: x['timestamp'])
        
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """验证数据质量"""
        try:
            if not data.get('data'):
                return False
                
            # 检查数据类型
            data_items = data['data']
            if not data_items:
                return False
                
            # 基本数据结构检查
            first_item = data_items[0]
            if isinstance(first_item, (MarketDataPoint, OHLCV)):
                return True
            elif isinstance(first_item, dict) and 'symbol' in first_item:
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"数据验证失败: {e}")
            return False
            
    def get_supported_symbols(self) -> List[str]:
        """获取支持的标的列表"""
        return self.supported_symbols
        
    def get_rate_limit(self) -> Dict[str, int]:
        """获取速率限制信息"""
        return {
            'requests_per_minute': 5,   # 免费版限制
            'requests_per_day': 500,    # 每日限制
            'concurrent_requests': 1    # 单线程访问
        }
        
    async def _perform_health_check(self) -> bool:
        """执行健康检查"""
        try:
            # 测试获取SPY报价
            test_data = await self._fetch_symbol_data('SPY', 'quote')
            return len(test_data) > 0
            
        except Exception as e:
            self.logger.error(f"Alpha Vantage健康检查失败: {e}")
            return False
            
    async def get_market_overview(self) -> Dict[str, Any]:
        """获取市场概览"""
        key_symbols = ['SPY', 'QQQ', 'IWM', 'VIX']
        return await self.fetch_data(key_symbols, data_type='quote')
        
    async def get_technical_analysis(
        self, 
        symbol: str, 
        indicators: List[str] = None
    ) -> Dict[str, Any]:
        """获取技术分析数据"""
        if indicators is None:
            indicators = ['sma', 'ema', 'rsi', 'macd']
            
        results = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'indicators': {}
        }
        
        for indicator in indicators:
            try:
                await asyncio.sleep(12)  # 频率限制
                data = await self._fetch_symbol_data(symbol, indicator)
                results['indicators'][indicator] = data
                
            except Exception as e:
                self.logger.error(f"获取技术指标失败 {indicator}: {e}")
                
        return results
        
    async def get_price_history(
        self, 
        symbol: str, 
        period: str = 'daily',
        days: int = 100
    ) -> Dict[str, Any]:
        """获取价格历史数据"""
        return await self.fetch_data(
            [symbol], 
            data_type=period,
            outputsize='compact' if days <= 100 else 'full'
        ) 