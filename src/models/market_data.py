"""市场数据模型"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class MarketDataType(str, Enum):
    """市场数据类型枚举"""
    PRICE = "price"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    OPTION_CHAIN = "option_chain"
    ECONOMIC_DATA = "economic_data"


class AssetType(str, Enum):
    """资产类型枚举"""
    STOCK = "stock"
    FUTURE = "future"
    INDEX = "index"
    ETF = "etf"
    OPTION = "option"
    BOND = "bond"
    FOREX = "forex"
    CRYPTO = "crypto"


class MarketDataPoint(BaseModel):
    """市场数据点"""
    timestamp: datetime = Field(description="时间戳")
    symbol: str = Field(description="标的代码")
    price: float = Field(description="价格")
    volume: Optional[float] = Field(default=None, description="交易量")
    bid: Optional[float] = Field(default=None, description="买价")
    ask: Optional[float] = Field(default=None, description="卖价")
    change: Optional[float] = Field(default=None, description="变化")
    change_percent: Optional[float] = Field(default=None, description="变化百分比")
    
    @validator('price')
    def validate_price(cls, v):
        if v < 0:
            raise ValueError('价格不能为负')
        return v


class OHLCV(BaseModel):
    """OHLCV数据"""
    timestamp: datetime = Field(description="时间戳")
    open: float = Field(description="开盘价")
    high: float = Field(description="最高价")
    low: float = Field(description="最低价")
    close: float = Field(description="收盘价")
    volume: float = Field(description="交易量")
    
    @validator('high')
    def validate_high(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('最高价不能低于最低价')
        return v


class MarketIndex(BaseModel):
    """市场指数"""
    symbol: str = Field(description="指数代码")
    name: str = Field(description="指数名称")
    current_price: float = Field(description="当前价格")
    change: float = Field(description="变化")
    change_percent: float = Field(description="变化百分比")
    volume: Optional[float] = Field(default=None, description="交易量")
    market_cap: Optional[float] = Field(default=None, description="市值")
    updated_at: datetime = Field(description="更新时间")


class FutureContract(BaseModel):
    """期货合约"""
    symbol: str = Field(description="合约代码")
    underlying: str = Field(description="标的资产")
    expiration_date: datetime = Field(description="到期日")
    contract_size: float = Field(description="合约规模")
    current_price: float = Field(description="当前价格")
    settlement_price: Optional[float] = Field(default=None, description="结算价")
    open_interest: Optional[float] = Field(default=None, description="持仓量")
    volume: Optional[float] = Field(default=None, description="交易量")
    margin_requirement: Optional[float] = Field(default=None, description="保证金要求")
    tick_size: Optional[float] = Field(default=None, description="最小变动价位")


class VolatilityData(BaseModel):
    """波动率数据"""
    symbol: str = Field(description="标的代码")
    implied_volatility: float = Field(description="隐含波动率")
    historical_volatility: float = Field(description="历史波动率")
    volatility_percentile: Optional[float] = Field(default=None, description="波动率百分位")
    vix_level: Optional[float] = Field(default=None, description="VIX水平")
    term_structure: Optional[Dict[str, float]] = Field(default=None, description="期限结构")
    skew: Optional[float] = Field(default=None, description="波动率偏斜")
    
    @validator('implied_volatility', 'historical_volatility')
    def validate_volatility(cls, v):
        if v < 0:
            raise ValueError('波动率不能为负')
        return v


class MarketSentiment(BaseModel):
    """市场情绪数据"""
    timestamp: datetime = Field(description="时间戳")
    fear_greed_index: Optional[float] = Field(default=None, description="恐慌贪婪指数")
    put_call_ratio: Optional[float] = Field(default=None, description="看跌看涨比率")
    vix_level: Optional[float] = Field(default=None, description="VIX水平")
    advance_decline_ratio: Optional[float] = Field(default=None, description="涨跌比率")
    high_low_index: Optional[float] = Field(default=None, description="新高新低指数")
    sentiment_score: Optional[float] = Field(default=None, description="综合情绪评分")
    
    @validator('sentiment_score')
    def validate_sentiment_score(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('情绪评分必须在0-100之间')
        return v


class MarketSession(BaseModel):
    """交易时段"""
    name: str = Field(description="时段名称")
    start_time: datetime = Field(description="开始时间")
    end_time: datetime = Field(description="结束时间")
    is_active: bool = Field(description="是否活跃")
    timezone: str = Field(description="时区")


class MarketSnapshot(BaseModel):
    """市场快照"""
    timestamp: datetime = Field(description="快照时间")
    session: MarketSession = Field(description="交易时段")
    indices: List[MarketIndex] = Field(description="主要指数")
    futures: List[FutureContract] = Field(description="期货合约")
    volatility: VolatilityData = Field(description="波动率数据")
    sentiment: MarketSentiment = Field(description="市场情绪")
    top_movers: Optional[List[MarketDataPoint]] = Field(default=None, description="涨跌幅榜")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MarketDataCollection(BaseModel):
    """市场数据集合"""
    source: str = Field(description="数据源")
    data_type: MarketDataType = Field(description="数据类型")
    asset_type: AssetType = Field(description="资产类型")
    symbols: List[str] = Field(description="标的代码列表")
    data: List[Union[MarketDataPoint, OHLCV, MarketIndex, FutureContract]] = Field(description="数据列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    collected_at: datetime = Field(default_factory=datetime.now, description="收集时间")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        
    def get_symbols(self) -> List[str]:
        """获取所有标的代码"""
        return list(set(self.symbols))
        
    def filter_by_symbol(self, symbol: str) -> List[Any]:
        """根据标的代码过滤数据"""
        return [item for item in self.data if hasattr(item, 'symbol') and item.symbol == symbol]
        
    def get_latest_data(self) -> Optional[Any]:
        """获取最新数据"""
        if not self.data:
            return None
        return max(self.data, key=lambda x: x.timestamp if hasattr(x, 'timestamp') else datetime.min) 