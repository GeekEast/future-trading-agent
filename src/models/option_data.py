"""期权数据模型"""

from datetime import datetime, date
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
import math


class OptionType(str, Enum):
    """期权类型枚举"""
    CALL = "call"
    PUT = "put"


class OptionStyle(str, Enum):
    """期权行权方式枚举"""
    AMERICAN = "american"
    EUROPEAN = "european"


class Option(BaseModel):
    """单个期权合约"""
    symbol: str = Field(description="期权代码")
    underlying: str = Field(description="标的代码")
    option_type: OptionType = Field(description="期权类型")
    strike: float = Field(description="行权价")
    expiration_date: date = Field(description="到期日")
    style: OptionStyle = Field(default=OptionStyle.AMERICAN, description="行权方式")
    
    # 市场数据
    last_price: Optional[float] = Field(default=None, description="最新价格")
    bid: Optional[float] = Field(default=None, description="买价")
    ask: Optional[float] = Field(default=None, description="卖价")
    volume: Optional[int] = Field(default=None, description="成交量")
    open_interest: Optional[int] = Field(default=None, description="持仓量")
    
    # 希腊字母
    delta: Optional[float] = Field(default=None, description="Delta")
    gamma: Optional[float] = Field(default=None, description="Gamma")
    theta: Optional[float] = Field(default=None, description="Theta")
    vega: Optional[float] = Field(default=None, description="Vega")
    rho: Optional[float] = Field(default=None, description="Rho")
    
    # 波动率
    implied_volatility: Optional[float] = Field(default=None, description="隐含波动率")
    
    @validator('strike')
    def validate_strike(cls, v):
        if v <= 0:
            raise ValueError('行权价必须大于0')
        return v
        
    @validator('implied_volatility')
    def validate_iv(cls, v):
        if v is not None and v < 0:
            raise ValueError('隐含波动率不能为负')
        return v
        
    def days_to_expiration(self, current_date: Optional[date] = None) -> int:
        """计算到期天数"""
        if current_date is None:
            current_date = date.today()
        return (self.expiration_date - current_date).days
        
    def time_to_expiration(self, current_date: Optional[date] = None) -> float:
        """计算到期时间（年化）"""
        days = self.days_to_expiration(current_date)
        return days / 365.0
        
    def is_itm(self, underlying_price: float) -> bool:
        """判断是否为实值期权"""
        if self.option_type == OptionType.CALL:
            return underlying_price > self.strike
        else:
            return underlying_price < self.strike
            
    def intrinsic_value(self, underlying_price: float) -> float:
        """计算内在价值"""
        if self.option_type == OptionType.CALL:
            return max(0, underlying_price - self.strike)
        else:
            return max(0, self.strike - underlying_price)
            
    def time_value(self, underlying_price: float) -> float:
        """计算时间价值"""
        if self.last_price is None:
            return 0
        return self.last_price - self.intrinsic_value(underlying_price)


class OptionChain(BaseModel):
    """期权链"""
    underlying: str = Field(description="标的代码")
    underlying_price: float = Field(description="标的价格")
    expiration_date: date = Field(description="到期日")
    options: List[Option] = Field(description="期权列表")
    timestamp: datetime = Field(default_factory=datetime.now, description="数据时间戳")
    
    def get_calls(self) -> List[Option]:
        """获取看涨期权"""
        return [opt for opt in self.options if opt.option_type == OptionType.CALL]
        
    def get_puts(self) -> List[Option]:
        """获取看跌期权"""
        return [opt for opt in self.options if opt.option_type == OptionType.PUT]
        
    def get_strikes(self) -> List[float]:
        """获取所有行权价"""
        return sorted(list(set(opt.strike for opt in self.options)))
        
    def get_itm_options(self) -> List[Option]:
        """获取实值期权"""
        return [opt for opt in self.options if opt.is_itm(self.underlying_price)]
        
    def get_otm_options(self) -> List[Option]:
        """获取虚值期权"""
        return [opt for opt in self.options if not opt.is_itm(self.underlying_price)]


class MaxPainAnalysis(BaseModel):
    """最大痛点分析"""
    underlying: str = Field(description="标的代码")
    expiration_date: date = Field(description="到期日")
    max_pain_price: float = Field(description="最大痛点价格")
    total_pain_at_max: float = Field(description="最大痛点处的总痛苦值")
    pain_by_strike: Dict[float, float] = Field(description="各行权价的痛苦值")
    call_put_ratio: float = Field(description="看涨看跌比率")
    
    def get_nearest_strikes(self, current_price: float, count: int = 5) -> List[float]:
        """获取最接近当前价格的行权价"""
        strikes = list(self.pain_by_strike.keys())
        strikes.sort(key=lambda x: abs(x - current_price))
        return strikes[:count]


class GammaExposure(BaseModel):
    """Gamma敞口分析"""
    underlying: str = Field(description="标的代码")
    underlying_price: float = Field(description="标的价格")
    net_gamma: float = Field(description="净Gamma")
    gamma_by_strike: Dict[float, float] = Field(description="各行权价的Gamma")
    zero_gamma_level: Optional[float] = Field(default=None, description="零Gamma水平")
    positive_gamma_range: Optional[List[float]] = Field(default=None, description="正Gamma区间")
    negative_gamma_range: Optional[List[float]] = Field(default=None, description="负Gamma区间")
    
    def is_positive_gamma_environment(self) -> bool:
        """判断是否为正Gamma环境"""
        return self.net_gamma > 0
        
    def get_gamma_flip_level(self) -> Optional[float]:
        """获取Gamma翻转水平"""
        return self.zero_gamma_level


class SupportResistance(BaseModel):
    """支撑阻力分析"""
    underlying: str = Field(description="标的代码")
    support_levels: List[float] = Field(description="支撑位")
    resistance_levels: List[float] = Field(description="阻力位")
    call_wall: Optional[float] = Field(default=None, description="Call墙")
    put_wall: Optional[float] = Field(default=None, description="Put墙")
    strength_scores: Dict[float, float] = Field(description="各价位强度评分")
    
    def get_nearest_support(self, current_price: float) -> Optional[float]:
        """获取最近的支撑位"""
        supports_below = [s for s in self.support_levels if s < current_price]
        return max(supports_below) if supports_below else None
        
    def get_nearest_resistance(self, current_price: float) -> Optional[float]:
        """获取最近的阻力位"""
        resistances_above = [r for r in self.resistance_levels if r > current_price]
        return min(resistances_above) if resistances_above else None


class VolatilitySurface(BaseModel):
    """波动率曲面"""
    underlying: str = Field(description="标的代码")
    surface_data: Dict[str, Dict[float, float]] = Field(description="波动率曲面数据")
    term_structure: Dict[str, float] = Field(description="期限结构")
    skew_by_expiry: Dict[str, float] = Field(description="各到期日的偏斜")
    timestamp: datetime = Field(default_factory=datetime.now, description="数据时间戳")
    
    def get_atm_volatility(self, expiry: str) -> Optional[float]:
        """获取平值波动率"""
        if expiry not in self.term_structure:
            return None
        return self.term_structure[expiry]
        
    def get_volatility_smile(self, expiry: str) -> Dict[float, float]:
        """获取波动率微笑"""
        return self.surface_data.get(expiry, {})


class OptionFlow(BaseModel):
    """期权流向数据"""
    timestamp: datetime = Field(description="时间戳")
    symbol: str = Field(description="期权代码")
    underlying: str = Field(description="标的代码")
    option_type: OptionType = Field(description="期权类型")
    strike: float = Field(description="行权价")
    expiration_date: date = Field(description="到期日")
    volume: int = Field(description="成交量")
    premium: float = Field(description="权利金")
    side: str = Field(description="买卖方向")
    is_opening: bool = Field(description="是否为开仓")
    
    def is_bullish(self) -> bool:
        """判断是否为看涨流向"""
        if self.option_type == OptionType.CALL:
            return self.side == "buy"
        else:
            return self.side == "sell"


class OptionMetrics(BaseModel):
    """期权市场指标"""
    underlying: str = Field(description="标的代码")
    timestamp: datetime = Field(description="时间戳")
    
    # 基础指标
    put_call_ratio: float = Field(description="看跌看涨比率")
    put_call_volume_ratio: float = Field(description="看跌看涨成交量比率")
    put_call_oi_ratio: float = Field(description="看跌看涨持仓比率")
    
    # 波动率指标
    vix: Optional[float] = Field(default=None, description="VIX指数")
    vvix: Optional[float] = Field(default=None, description="VVIX指数")
    realized_volatility: Optional[float] = Field(default=None, description="已实现波动率")
    iv_rank: Optional[float] = Field(default=None, description="隐含波动率排名")
    iv_percentile: Optional[float] = Field(default=None, description="隐含波动率百分位")
    
    # 期权结构指标
    max_pain: Optional[float] = Field(default=None, description="最大痛点")
    gamma_exposure: Optional[float] = Field(default=None, description="Gamma敞口")
    dealer_gamma_position: Optional[float] = Field(default=None, description="做市商Gamma头寸")
    
    def get_sentiment_score(self) -> float:
        """计算期权情绪评分"""
        score = 50  # 中性
        
        # Put/Call比率影响
        if self.put_call_ratio > 1.2:
            score -= 20  # 过度悲观
        elif self.put_call_ratio < 0.8:
            score += 20  # 过度乐观
            
        # VIX影响
        if self.vix is not None:
            if self.vix > 30:
                score -= 15  # 恐慌
            elif self.vix < 15:
                score += 15  # 自满
                
        return max(0, min(100, score))


class OptionAnalysisResult(BaseModel):
    """期权分析结果"""
    underlying: str = Field(description="标的代码")
    underlying_price: float = Field(description="标的价格")
    analysis_time: datetime = Field(default_factory=datetime.now, description="分析时间")
    
    # 分析结果
    option_chains: List[OptionChain] = Field(description="期权链")
    max_pain: MaxPainAnalysis = Field(description="最大痛点分析")
    gamma_exposure: GammaExposure = Field(description="Gamma敞口分析")
    support_resistance: SupportResistance = Field(description="支撑阻力分析")
    volatility_surface: VolatilitySurface = Field(description="波动率曲面")
    metrics: OptionMetrics = Field(description="期权指标")
    
    # 交易建议
    trading_range: Optional[List[float]] = Field(default=None, description="预期交易区间")
    risk_level: str = Field(description="风险等级")
    market_outlook: str = Field(description="市场展望")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }
        
    def get_key_levels(self) -> Dict[str, float]:
        """获取关键价位"""
        levels = {}
        
        if self.max_pain:
            levels['max_pain'] = self.max_pain.max_pain_price
            
        if self.support_resistance.call_wall:
            levels['call_wall'] = self.support_resistance.call_wall
            
        if self.support_resistance.put_wall:
            levels['put_wall'] = self.support_resistance.put_wall
            
        if self.gamma_exposure.zero_gamma_level:
            levels['zero_gamma'] = self.gamma_exposure.zero_gamma_level
            
        return levels
        
    def get_trading_summary(self) -> Dict[str, Any]:
        """获取交易摘要"""
        return {
            'underlying': self.underlying,
            'current_price': self.underlying_price,
            'key_levels': self.get_key_levels(),
            'gamma_environment': 'positive' if self.gamma_exposure.is_positive_gamma_environment() else 'negative',
            'sentiment_score': self.metrics.get_sentiment_score(),
            'risk_level': self.risk_level,
            'outlook': self.market_outlook
        } 