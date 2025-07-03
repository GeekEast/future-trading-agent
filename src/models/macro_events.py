"""宏观经济事件和数据模型"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class EconomicIndicatorType(str, Enum):
    """经济指标类型枚举"""
    EMPLOYMENT = "employment"
    INFLATION = "inflation"
    GROWTH = "growth"
    MONETARY = "monetary"
    TRADE = "trade"
    CONSUMER = "consumer"
    MANUFACTURING = "manufacturing"
    HOUSING = "housing"
    SENTIMENT = "sentiment"


class EventImportance(str, Enum):
    """事件重要性枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyStance(str, Enum):
    """政策立场枚举"""
    HAWKISH = "hawkish"      # 鹰派
    DOVISH = "dovish"        # 鸽派
    NEUTRAL = "neutral"      # 中性
    MIXED = "mixed"          # 混合


class FedPolicyAction(str, Enum):
    """美联储政策行动枚举"""
    RATE_HIKE = "rate_hike"
    RATE_CUT = "rate_cut"
    HOLD = "hold"
    QE = "quantitative_easing"
    QT = "quantitative_tightening"
    GUIDANCE = "forward_guidance"


class EconomicIndicator(BaseModel):
    """经济指标"""
    name: str = Field(description="指标名称")
    symbol: str = Field(description="指标代码")
    value: float = Field(description="指标值")
    unit: str = Field(description="单位")
    frequency: str = Field(description="发布频率")
    release_date: datetime = Field(description="发布日期")
    previous_value: Optional[float] = Field(default=None, description="前值")
    forecast_value: Optional[float] = Field(default=None, description="预测值")
    revision: Optional[float] = Field(default=None, description="修正值")
    importance: EventImportance = Field(description="重要性")
    indicator_type: EconomicIndicatorType = Field(description="指标类型")
    
    def get_surprise(self) -> Optional[float]:
        """计算超预期幅度"""
        if self.forecast_value is None:
            return None
        return self.value - self.forecast_value
        
    def get_change_from_previous(self) -> Optional[float]:
        """计算与前值的变化"""
        if self.previous_value is None:
            return None
        return self.value - self.previous_value


class FedOfficial(BaseModel):
    """美联储官员"""
    name: str = Field(description="姓名")
    title: str = Field(description="职务")
    voting_member: bool = Field(description="是否为投票成员")
    hawkish_dovish: Optional[str] = Field(default=None, description="鹰派/鸽派倾向")


class FedSpeech(BaseModel):
    """美联储讲话"""
    official: FedOfficial = Field(description="讲话官员")
    title: str = Field(description="讲话标题")
    date: datetime = Field(description="讲话日期")
    location: str = Field(description="讲话地点")
    key_points: List[str] = Field(description="关键要点")
    market_impact: Optional[str] = Field(default=None, description="市场影响")
    hawkish_dovish_score: Optional[float] = Field(default=None, description="鹰派/鸽派评分")
    
    @validator('hawkish_dovish_score')
    def validate_score(cls, v):
        if v is not None and (v < -10 or v > 10):
            raise ValueError('鹰派/鸽派评分必须在-10到10之间')
        return v


class FOMCMeeting(BaseModel):
    """FOMC会议"""
    date: datetime = Field(description="会议日期")
    decision: FedPolicyAction = Field(description="政策决定")
    target_rate: float = Field(description="目标利率")
    rate_change: float = Field(description="利率变化")
    vote_count: Dict[str, int] = Field(description="投票情况")
    statement_highlights: List[str] = Field(description="声明要点")
    dot_plot: Optional[Dict[str, List[float]]] = Field(default=None, description="点阵图")
    press_conference: Optional[bool] = Field(default=None, description="是否有记者会")
    
    @validator('target_rate')
    def validate_rate(cls, v):
        if v < 0 or v > 25:
            raise ValueError('目标利率必须在0-25%之间')
        return v


class FedWatchData(BaseModel):
    """CME FedWatch数据"""
    date: datetime = Field(description="数据日期")
    meeting_date: datetime = Field(description="会议日期")
    probabilities: Dict[str, float] = Field(description="概率分布")
    expected_rate: float = Field(description="预期利率")
    change_from_previous: float = Field(description="与前次数据的变化")
    
    @validator('probabilities')
    def validate_probabilities(cls, v):
        total = sum(v.values())
        if not (0.95 <= total <= 1.05):  # 允许小的舍入误差
            raise ValueError('概率总和必须接近1')
        return v


class EconomicEvent(BaseModel):
    """经济事件"""
    title: str = Field(description="事件标题")
    description: str = Field(description="事件描述")
    date: datetime = Field(description="事件日期")
    country: str = Field(description="国家")
    category: str = Field(description="类别")
    importance: EventImportance = Field(description="重要性")
    actual_value: Optional[float] = Field(default=None, description="实际值")
    forecast_value: Optional[float] = Field(default=None, description="预测值")
    previous_value: Optional[float] = Field(default=None, description="前值")
    unit: Optional[str] = Field(default=None, description="单位")
    source: str = Field(description="数据源")
    
    def is_surprise(self) -> bool:
        """判断是否超预期"""
        if self.actual_value is None or self.forecast_value is None:
            return False
        return abs(self.actual_value - self.forecast_value) > 0.1  # 可配置阈值


class MarketImpactAssessment(BaseModel):
    """市场影响评估"""
    event: Union[EconomicEvent, EconomicIndicator, FOMCMeeting] = Field(description="经济事件")
    impact_score: float = Field(description="影响评分")
    affected_markets: List[str] = Field(description="受影响市场")
    direction: str = Field(description="影响方向")
    duration: str = Field(description="影响持续时间")
    confidence: float = Field(description="评估置信度")
    reasoning: str = Field(description="分析理由")
    
    @validator('impact_score')
    def validate_impact_score(cls, v):
        if v < 0 or v > 10:
            raise ValueError('影响评分必须在0-10之间')
        return v
        
    @validator('confidence')
    def validate_confidence(cls, v):
        if v < 0 or v > 1:
            raise ValueError('置信度必须在0-1之间')
        return v


class MacroEnvironment(BaseModel):
    """宏观环境"""
    timestamp: datetime = Field(description="时间戳")
    gdp_growth: Optional[float] = Field(default=None, description="GDP增长率")
    inflation_rate: Optional[float] = Field(default=None, description="通胀率")
    unemployment_rate: Optional[float] = Field(default=None, description="失业率")
    interest_rate: Optional[float] = Field(default=None, description="利率")
    yield_curve_slope: Optional[float] = Field(default=None, description="收益率曲线斜率")
    dollar_index: Optional[float] = Field(default=None, description="美元指数")
    commodity_index: Optional[float] = Field(default=None, description="商品指数")
    
    def get_economic_cycle_phase(self) -> str:
        """判断经济周期阶段"""
        if self.gdp_growth is None or self.inflation_rate is None:
            return "unknown"
            
        if self.gdp_growth > 2.5 and self.inflation_rate < 3.0:
            return "expansion"
        elif self.gdp_growth < 0:
            return "recession"
        elif self.inflation_rate > 4.0:
            return "stagflation"
        else:
            return "slowdown"


class MacroDataCollection(BaseModel):
    """宏观数据集合"""
    source: str = Field(description="数据源")
    collected_at: datetime = Field(default_factory=datetime.now, description="收集时间")
    indicators: List[EconomicIndicator] = Field(description="经济指标")
    events: List[EconomicEvent] = Field(description="经济事件")
    fed_data: Dict[str, Any] = Field(description="美联储数据")
    impact_assessments: List[MarketImpactAssessment] = Field(description="市场影响评估")
    macro_environment: MacroEnvironment = Field(description="宏观环境")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        
    def get_high_impact_events(self) -> List[EconomicEvent]:
        """获取高影响事件"""
        return [event for event in self.events if event.importance in [EventImportance.HIGH, EventImportance.CRITICAL]]
        
    def get_surprises(self) -> List[EconomicEvent]:
        """获取超预期事件"""
        return [event for event in self.events if event.is_surprise()]
        
    def get_upcoming_events(self, days: int = 7) -> List[EconomicEvent]:
        """获取未来几天的事件"""
        from datetime import timedelta
        cutoff_date = datetime.now() + timedelta(days=days)
        return [event for event in self.events if event.date <= cutoff_date and event.date >= datetime.now()] 