"""Yahoo Finance数据源 - 免费市场和期权数据"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import yfinance as yf
import pandas as pd

from .base_source import BaseDataSource
from models.market_data import MarketDataPoint, OHLCV, MarketIndex, AssetType
from models.option_data import Option, OptionChain, OptionType, OptionStyle
from utils.helpers import convert_to_numeric, ValidationError


class YahooFinanceDataSource(BaseDataSource):
    """Yahoo Finance 免费数据源"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("YahooFinance", config)
        
        # 支持的全球市场指数
        self.global_indices = {
            # 美国主要指数
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^RUT': 'Russell 2000',
            '^VIX': 'VIX',
            
            # 全球主要指数
            '^FTSE': 'FTSE 100',
            '^GDAXI': 'DAX',
            '^FCHI': 'CAC 40',
            '^N225': 'Nikkei 225',
            '^HSI': 'Hang Seng',
            '000001.SS': 'Shanghai Composite',
            
            # 债券
            '^TNX': '10-Year Treasury',
            '^FVX': '5-Year Treasury',
            '^TYX': '30-Year Treasury',
            
            # 商品相关ETF
            'GLD': 'Gold ETF',
            'SLV': 'Silver ETF',
            'USO': 'Oil ETF',
            'UNG': 'Natural Gas ETF',
            'DBA': 'Agriculture ETF'
        }
        
        # 期权活跃的股票
        self.option_symbols = [
            'SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 
            'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'
        ]
        
    async def _custom_initialize(self) -> None:
        """Yahoo Finance不需要API密钥"""
        self.logger.info("Yahoo Finance数据源初始化完成")
        
    async def fetch_data(
        self, 
        symbols: List[str], 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """获取市场数据"""
        
        data_type = kwargs.get('data_type', 'quote')
        period = kwargs.get('period', '1y')
        interval = kwargs.get('interval', '1d')
        
        results = {
            'source': 'YahooFinance',
            'timestamp': datetime.now(),
            'data_type': data_type,
            'data': [],
            'metadata': {
                'symbols_requested': symbols,
                'period': period,
                'interval': interval
            }
        }
        
        for symbol in symbols:
            try:
                if data_type == 'quote':
                    data = await self._fetch_quote_data(symbol)
                elif data_type == 'history':
                    data = await self._fetch_history_data(symbol, period, interval)
                elif data_type == 'options':
                    data = await self._fetch_option_data(symbol)
                elif data_type == 'info':
                    data = await self._fetch_info_data(symbol)
                else:
                    self.logger.warning(f"不支持的数据类型: {data_type}")
                    continue
                    
                if data:
                    results['data'].extend(data)
                    
            except Exception as e:
                self.logger.error(f"获取Yahoo Finance数据失败 {symbol}: {e}")
                continue
                
        return results
        
    async def _fetch_quote_data(self, symbol: str) -> List[MarketDataPoint]:
        """获取实时报价数据"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return []
                
            # 尝试不同的价格字段
            price = convert_to_numeric(info.get('currentPrice', 0)) or \
                   convert_to_numeric(info.get('regularMarketPrice', 0)) or \
                   convert_to_numeric(info.get('previousClose', 0))
            
            if price <= 0:
                # 跳过无效价格
                return []
                
            data_point = MarketDataPoint(
                timestamp=datetime.now(),
                symbol=symbol,
                price=price,
                volume=convert_to_numeric(info.get('volume', 0)) or convert_to_numeric(info.get('regularMarketVolume', 0)),
                bid=convert_to_numeric(info.get('bid', 0)),
                ask=convert_to_numeric(info.get('ask', 0)),
                change=convert_to_numeric(info.get('regularMarketChange', 0)),
                change_percent=convert_to_numeric(info.get('regularMarketChangePercent', 0)) * 100
            )
            
            return [data_point]
            
        except Exception as e:
            self.logger.error(f"获取报价数据失败 {symbol}: {e}")
            return []
            
    async def _fetch_history_data(
        self, 
        symbol: str, 
        period: str, 
        interval: str
    ) -> List[OHLCV]:
        """获取历史数据"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                return []
                
            ohlcv_data = []
            for index, row in hist.iterrows():
                ohlcv = OHLCV(
                    timestamp=index.to_pydatetime(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row['Volume'])
                )
                ohlcv_data.append(ohlcv)
                
            return ohlcv_data
            
        except Exception as e:
            self.logger.error(f"获取历史数据失败 {symbol}: {e}")
            return []
            
    async def _fetch_option_data(self, symbol: str) -> List[OptionChain]:
        """获取期权链数据"""
        try:
            ticker = yf.Ticker(symbol)
            
            # 获取期权到期日
            expiration_dates = ticker.options
            if not expiration_dates:
                return []
                
            option_chains = []
            current_price = await self._get_current_price(symbol)
            
            # 只获取前3个到期日，避免数据过多
            for exp_date in expiration_dates[:3]:
                try:
                    option_chain = ticker.option_chain(exp_date)
                    
                    # 解析看涨期权
                    calls = self._parse_option_data(
                        option_chain.calls, symbol, exp_date, OptionType.CALL
                    )
                    
                    # 解析看跌期权
                    puts = self._parse_option_data(
                        option_chain.puts, symbol, exp_date, OptionType.PUT
                    )
                    
                    # 创建期权链对象
                    chain = OptionChain(
                        underlying=symbol,
                        underlying_price=current_price,
                        expiration_date=datetime.strptime(exp_date, '%Y-%m-%d').date(),
                        options=calls + puts
                    )
                    
                    option_chains.append(chain)
                    
                except Exception as e:
                    self.logger.warning(f"解析期权链失败 {symbol} {exp_date}: {e}")
                    continue
                    
            return option_chains
            
        except Exception as e:
            self.logger.error(f"获取期权数据失败 {symbol}: {e}")
            return []
            
    def _parse_option_data(
        self, 
        options_df: pd.DataFrame, 
        symbol: str, 
        exp_date: str, 
        option_type: OptionType
    ) -> List[Option]:
        """解析期权数据"""
        options = []
        
        for _, row in options_df.iterrows():
            try:
                option = Option(
                    symbol=row.get('contractSymbol', ''),
                    underlying=symbol,
                    option_type=option_type,
                    strike=float(row.get('strike', 0)),
                    expiration_date=datetime.strptime(exp_date, '%Y-%m-%d').date(),
                    style=OptionStyle.AMERICAN,
                    last_price=convert_to_numeric(row.get('lastPrice', 0)),
                    bid=convert_to_numeric(row.get('bid', 0)),
                    ask=convert_to_numeric(row.get('ask', 0)),
                    volume=int(convert_to_numeric(row.get('volume', 0))),
                    open_interest=int(convert_to_numeric(row.get('openInterest', 0))),
                    implied_volatility=convert_to_numeric(row.get('impliedVolatility', 0))
                )
                
                options.append(option)
                
            except Exception as e:
                self.logger.warning(f"解析期权数据点失败: {e}")
                continue
                
        return options
        
    async def _get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return convert_to_numeric(info.get('currentPrice', 0))
            
        except Exception as e:
            self.logger.error(f"获取当前价格失败 {symbol}: {e}")
            return 0.0
            
    async def _fetch_info_data(self, symbol: str) -> List[Dict[str, Any]]:
        """获取股票基本信息"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return []
                
            return [{
                'symbol': symbol,
                'name': info.get('longName', ''),
                'market_cap': info.get('marketCap', 0),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0)
            }]
            
        except Exception as e:
            self.logger.error(f"获取股票信息失败 {symbol}: {e}")
            return []
            
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """验证数据质量"""
        try:
            if not data.get('data'):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"数据验证失败: {e}")
            return False
            
    def get_supported_symbols(self) -> List[str]:
        """获取支持的标的列表"""
        return list(self.global_indices.keys()) + self.option_symbols
        
    def get_rate_limit(self) -> Dict[str, int]:
        """获取速率限制信息"""
        return {
            'requests_per_minute': 60,  # 相对宽松
            'requests_per_day': 2000,
            'concurrent_requests': 5
        }
        
    async def _perform_health_check(self) -> bool:
        """执行健康检查"""
        try:
            test_data = await self._fetch_quote_data('AAPL')
            return len(test_data) > 0
            
        except Exception as e:
            self.logger.error(f"Yahoo Finance健康检查失败: {e}")
            return False
            
    async def get_global_market_overview(self) -> Dict[str, Any]:
        """获取全球市场概览"""
        key_indices = ['^GSPC', '^DJI', '^IXIC', '^VIX', '^TNX']
        return await self.fetch_data(key_indices, data_type='quote')
        
    async def get_vix_data(self) -> Dict[str, Any]:
        """获取VIX波动率数据"""
        return await self.fetch_data(['^VIX', '^VVIX'], data_type='history', period='1mo')
        
    async def get_bond_yields(self) -> Dict[str, Any]:
        """获取债券收益率"""
        bond_symbols = ['^TNX', '^FVX', '^TYX', '^IRX']
        return await self.fetch_data(bond_symbols, data_type='quote')
        
    async def get_commodity_etfs(self) -> Dict[str, Any]:
        """获取商品ETF数据"""
        commodity_symbols = ['GLD', 'SLV', 'USO', 'UNG', 'DBA']
        return await self.fetch_data(commodity_symbols, data_type='quote')
        
    async def get_option_chain(self, symbol: str) -> Dict[str, Any]:
        """获取特定股票的期权链"""
        if symbol not in self.option_symbols:
            self.logger.warning(f"股票 {symbol} 可能没有活跃期权")
            
        return await self.fetch_data([symbol], data_type='options')
        
    async def get_market_sentiment_indicators(self) -> Dict[str, Any]:
        """获取市场情绪指标"""
        sentiment_symbols = ['^VIX', '^VVIX', 'UVXY', 'SVXY']
        return await self.fetch_data(sentiment_symbols, data_type='quote') 