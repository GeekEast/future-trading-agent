"""
全球市场情绪数据模型
支持VIX分析、跨资产相关性、地缘政治风险等
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from decimal import Decimal


class SentimentType(Enum):
    """市场情绪类型"""
    FEAR = "fear"           # 恐慌
    GREED = "greed"         # 贪婪
    NEUTRAL = "neutral"     # 中性
    UNCERTAINTY = "uncertainty"  # 不确定


class RiskLevel(Enum):
    """风险级别"""
    LOW = "low"             # 低风险
    MEDIUM = "medium"       # 中等风险
    HIGH = "high"           # 高风险
    EXTREME = "extreme"     # 极端风险
    
    def __lt__(self, other):
        """定义小于比较"""
        order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.EXTREME]
        return order.index(self) < order.index(other)
        
    def __le__(self, other):
        """定义小于等于比较"""
        return self == other or self < other
        
    def __gt__(self, other):
        """定义大于比较"""
        return not self <= other
        
    def __ge__(self, other):
        """定义大于等于比较"""
        return self == other or self > other


class AssetClass(Enum):
    """资产类别"""
    EQUITY = "equity"       # 股票
    BOND = "bond"          # 债券
    COMMODITY = "commodity" # 商品
    CURRENCY = "currency"   # 货币
    CRYPTO = "crypto"      # 加密货币
    VOLATILITY = "volatility" # 波动率


@dataclass
class GlobalIndex:
    """全球指数数据"""
    symbol: str
    name: str
    value: float
    change: float
    change_percent: float
    volume: Optional[float]
    market_cap: Optional[float]
    region: str  # US, EU, ASIA, etc.
    sector: Optional[str]
    timestamp: datetime


@dataclass
class VolatilityIndicator:
    """波动率指标"""
    symbol: str  # VIX, VSTOXX, VVIX等
    name: str
    value: float
    change: float
    change_percent: float
    percentile_rank: float  # 历史百分位
    interpretation: str  # 解读说明
    timestamp: datetime


@dataclass
class CrossAssetCorrelation:
    """跨资产相关性"""
    asset1: str
    asset2: str
    asset1_class: AssetClass
    asset2_class: AssetClass
    correlation: float  # -1 to 1
    period_days: int    # 计算周期
    significance: float  # 统计显著性
    interpretation: str
    timestamp: datetime


@dataclass
class GeopoliticalEvent:
    """地缘政治事件"""
    event_id: str
    title: str
    description: str
    country: str
    region: str
    severity: RiskLevel
    market_impact: str  # 对市场的预期影响
    affected_assets: List[str]
    probability: float  # 事件发生概率
    start_date: datetime
    end_date: Optional[datetime]
    sources: List[str]


@dataclass
class NewsEventSentiment:
    """新闻事件情绪"""
    headline: str
    content: str
    source: str
    sentiment_score: float  # -1 (negative) to 1 (positive)
    importance: float      # 0 to 1
    market_relevance: float # 0 to 1
    affected_instruments: List[str]
    keywords: List[str]
    published_time: datetime
    analyzed_time: datetime


@dataclass
class MarketRegimeIndicator:
    """市场制度指标"""
    regime_type: str    # bull_market, bear_market, sideways, crisis
    confidence: float   # 0 to 1
    duration_days: int  # 当前制度持续天数
    typical_duration: int # 典型持续时间
    key_drivers: List[str]
    exit_signals: List[str]
    historical_precedents: List[str]
    timestamp: datetime


@dataclass
class FlightToQualitySignal:
    """避险情绪信号"""
    signal_strength: float  # 0 to 1
    direction: str         # to_safety, from_safety
    safe_haven_flows: Dict[str, float]  # 资产 -> 流入量
    risk_asset_outflows: Dict[str, float]  # 资产 -> 流出量
    bond_yields: Dict[str, float]  # 各期限债券收益率
    gold_performance: float
    usd_strength: float
    timestamp: datetime


@dataclass
class TechnicalSentiment:
    """技术面情绪指标"""
    symbol: str
    rsi: float              # 相对强弱指标
    macd: float             # MACD
    bollinger_position: float # 布林带位置
    support_level: float
    resistance_level: float
    trend_strength: float   # 趋势强度
    momentum: float         # 动量
    volume_profile: str     # 成交量特征
    timestamp: datetime


@dataclass
class SentimentAnalysis:
    """综合情绪分析结果"""
    overall_sentiment: SentimentType
    sentiment_score: float  # -1 to 1
    confidence: float       # 0 to 1
    risk_level: RiskLevel
    
    # 分类情绪
    equity_sentiment: float
    bond_sentiment: float
    commodity_sentiment: float
    currency_sentiment: float
    
    # 关键指标
    fear_greed_index: float  # 0 (fear) to 100 (greed)
    volatility_regime: str   # low, normal, high, extreme
    correlation_regime: str  # normal, high, crisis
    
    # 主要驱动因素
    primary_drivers: List[str]
    risk_factors: List[str]
    opportunities: List[str]
    
    # 预测和建议
    short_term_outlook: str  # 1-7天
    medium_term_outlook: str # 1-4周
    recommended_positioning: List[str]
    hedge_suggestions: List[str]
    
    analysis_timestamp: datetime
    next_update: datetime


@dataclass
class SentimentReport:
    """情绪分析报告"""
    analysis: SentimentAnalysis
    global_indices: List[GlobalIndex]
    volatility_indicators: List[VolatilityIndicator]
    correlations: List[CrossAssetCorrelation]
    geopolitical_events: List[GeopoliticalEvent]
    news_sentiment: List[NewsEventSentiment]
    market_regime: MarketRegimeIndicator
    flight_to_quality: FlightToQualitySignal
    technical_sentiment: List[TechnicalSentiment]
    
    report_timestamp: datetime
    data_coverage: Dict[str, Any]  # 数据覆盖统计
    reliability_score: float        # 分析可靠性评分 