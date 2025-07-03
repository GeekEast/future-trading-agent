"""
综合分析数据模型
整合MacroPolicy、GlobalSentiment、OptionStructure三个Agent的分析结果
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from decimal import Decimal

# PolicyAnalysis is defined in agents.macro_policy_agent, import will be handled dynamically
from .sentiment_data import SentimentAnalysis
from .option_data import OptionAnalysisResult


class MarketDirection(Enum):
    """市场方向"""
    BULLISH = "bullish"         # 看涨
    BEARISH = "bearish"         # 看跌
    NEUTRAL = "neutral"         # 中性
    UNCERTAIN = "uncertain"     # 不确定


class ConflictResolution(Enum):
    """冲突解决策略"""
    MACRO_PRIORITY = "macro_priority"           # 宏观优先
    SENTIMENT_PRIORITY = "sentiment_priority"   # 情绪优先
    OPTION_PRIORITY = "option_priority"         # 期权优先
    CONSENSUS = "consensus"                     # 共识
    WEIGHTED_AVERAGE = "weighted_average"       # 加权平均


class TimeHorizon(Enum):
    """时间周期"""
    INTRADAY = "intraday"       # 日内
    SHORT_TERM = "short_term"   # 短期（1-7天）
    MEDIUM_TERM = "medium_term" # 中期（1-4周）
    LONG_TERM = "long_term"     # 长期（1-3月）


@dataclass
class AgentConsensus:
    """Agent共识度分析"""
    macro_sentiment_agreement: float    # 宏观与情绪一致性
    sentiment_option_agreement: float   # 情绪与期权一致性
    macro_option_agreement: float       # 宏观与期权一致性
    overall_consensus: float            # 总体共识度
    
    # 分歧点分析
    main_disagreements: List[str]       # 主要分歧
    consensus_areas: List[str]          # 共识领域
    
    def get_consensus_level(self) -> str:
        """获取共识级别"""
        if self.overall_consensus > 0.8:
            return "高度共识"
        elif self.overall_consensus > 0.6:
            return "基本共识"
        elif self.overall_consensus > 0.4:
            return "部分共识"
        else:
            return "分歧较大"


@dataclass
class MarketOutlook:
    """市场展望"""
    direction: MarketDirection
    confidence: float               # 置信度
    time_horizon: TimeHorizon
    
    # 分时间段预测
    intraday_outlook: str          # 日内展望
    short_term_outlook: str        # 短期展望
    medium_term_outlook: str       # 中期展望
    
    # 概率分布
    bullish_probability: float     # 看涨概率
    bearish_probability: float     # 看跌概率
    sideways_probability: float    # 横盘概率
    
    # 关键因素
    supporting_factors: List[str]  # 支撑因素
    risk_factors: List[str]        # 风险因素
    catalysts: List[str]           # 催化剂
    
    # 情景分析
    best_case_scenario: str        # 最佳情形
    worst_case_scenario: str       # 最坏情形
    base_case_scenario: str        # 基准情形


@dataclass
class TradingSignal:
    """交易信号"""
    signal_type: str               # 信号类型：BUY, SELL, HOLD
    strength: float                # 信号强度 (0-1)
    confidence: float              # 信号置信度 (0-1)
    time_horizon: TimeHorizon      # 适用时间周期
    
    # 具体建议
    entry_price: Optional[float]   # 入场价格
    target_price: Optional[float]  # 目标价格
    stop_loss: Optional[float]     # 止损价格
    position_size: Optional[float] # 建议仓位
    
    # 支撑信息
    supporting_agents: List[str]   # 支撑该信号的Agent
    contradicting_agents: List[str] # 反对该信号的Agent
    
    reasoning: str                 # 信号逻辑
    risk_reward_ratio: Optional[float] # 风险收益比


@dataclass
class RiskAssessment:
    """风险评估"""
    overall_risk_level: str        # 总体风险级别
    risk_score: float              # 风险评分 (0-100)
    
    # 分类风险
    macro_risk: float              # 宏观风险
    sentiment_risk: float          # 情绪风险
    technical_risk: float          # 技术风险
    liquidity_risk: float          # 流动性风险
    volatility_risk: float         # 波动率风险
    
    # 风险因素
    top_risks: List[str]           # 主要风险
    tail_risks: List[str]          # 尾部风险
    black_swan_events: List[str]   # 黑天鹅事件
    
    # 风险控制建议
    hedging_suggestions: List[str] # 对冲建议
    position_limits: Dict[str, float] # 仓位限制
    stop_loss_levels: Dict[str, float] # 止损水平
    
    # 可选字段必须放在最后
    systemic_risk: Any = None      # 系统性风险级别 (RiskLevel枚举)


@dataclass
class AllocationRecommendation:
    """配置建议"""
    recommended_allocation: Dict[str, float]  # 推荐配置
    
    # 资产类别配置
    equity_allocation: float       # 股票配置
    bond_allocation: float         # 债券配置
    commodity_allocation: float    # 商品配置
    cash_allocation: float         # 现金配置
    alternative_allocation: float  # 另类投资配置
    
    # 风险调整
    risk_parity: bool              # 是否风险平价
    leverage_recommendation: float # 杠杆建议
    
    # 动态调整策略
    rebalancing_frequency: str     # 再平衡频率
    trigger_conditions: List[str]  # 触发条件
    
    reasoning: str                 # 配置逻辑


@dataclass
class MarketRegimeAnalysis:
    """市场制度分析"""
    current_regime: str            # 当前市场制度
    regime_confidence: float       # 制度判断置信度
    regime_duration: int           # 制度持续天数
    
    # 制度特征
    volatility_regime: str         # 波动率制度
    correlation_regime: str        # 相关性制度
    liquidity_regime: str          # 流动性制度
    
    # 制度转换分析
    transition_probability: float  # 转换概率
    next_likely_regime: str        # 下一个可能制度
    transition_catalysts: List[str] # 转换催化剂
    
    # 历史类比
    historical_parallels: List[str] # 历史相似情况
    lessons_learned: List[str]     # 历史教训


@dataclass
class ComprehensiveAnalysis:
    """综合分析结果"""
    
    # 基本信息
    analysis_timestamp: datetime
    target_symbol: str
    analysis_version: str
    
    # 输入数据
    macro_analysis: Any  # PolicyAnalysis from agents.macro_policy_agent
    sentiment_analysis: SentimentAnalysis
    option_analysis: OptionAnalysisResult
    
    # 综合分析结果
    consensus: AgentConsensus
    market_outlook: MarketOutlook
    trading_signals: List[TradingSignal]
    risk_assessment: RiskAssessment
    allocation_recommendation: AllocationRecommendation
    regime_analysis: MarketRegimeAnalysis
    
    # 关键指标
    overall_score: float           # 总体评分
    signal_quality: float          # 信号质量
    data_completeness: float       # 数据完整性
    analysis_reliability: float    # 分析可靠性
    
    # 决策支持
    executive_summary: str         # 执行摘要
    key_takeaways: List[str]       # 关键要点
    action_items: List[str]        # 行动项
    
    # 更新计划
    next_update_time: datetime
    monitoring_points: List[str]   # 监控要点
    
    def get_primary_signal(self) -> Optional[TradingSignal]:
        """获取主要交易信号"""
        if not self.trading_signals:
            return None
        
        # 返回强度最高的信号
        return max(self.trading_signals, key=lambda x: x.strength * x.confidence)
    
    def get_risk_adjusted_score(self) -> float:
        """获取风险调整后评分"""
        risk_adjustment = (100 - self.risk_assessment.risk_score) / 100
        return self.overall_score * risk_adjustment
    
    def get_consensus_summary(self) -> str:
        """获取共识摘要"""
        level = self.consensus.get_consensus_level()
        if self.consensus.main_disagreements:
            disagreements = "、".join(self.consensus.main_disagreements[:2])
            return f"{level}，主要分歧在{disagreements}"
        else:
            return f"{level}，各Agent观点基本一致"
    
    def get_market_direction_summary(self) -> str:
        """获取市场方向摘要"""
        direction_map = {
            MarketDirection.BULLISH: "看涨",
            MarketDirection.BEARISH: "看跌",
            MarketDirection.NEUTRAL: "中性",
            MarketDirection.UNCERTAIN: "不确定"
        }
        
        direction_str = direction_map.get(self.market_outlook.direction, "未知")
        confidence_str = f"{self.market_outlook.confidence:.0%}"
        
        return f"{direction_str}（置信度{confidence_str}）"


@dataclass
class ComprehensiveReport:
    """综合分析报告"""
    analysis: ComprehensiveAnalysis
    report_timestamp: datetime
    report_format: str             # 报告格式：markdown, html, pdf
    
    # 报告内容
    executive_summary: str
    detailed_analysis: str
    trading_recommendations: str
    risk_disclosures: str
    appendix: str
    
    # 报告元数据
    report_length: int             # 报告长度
    charts_included: List[str]     # 包含的图表
    data_sources: List[str]        # 数据源
    
    def get_report_quality_score(self) -> float:
        """获取报告质量评分"""
        quality_factors = []
        
        # 数据完整性
        quality_factors.append(self.analysis.data_completeness)
        
        # 分析可靠性
        quality_factors.append(self.analysis.analysis_reliability)
        
        # 共识度
        quality_factors.append(self.analysis.consensus.overall_consensus)
        
        # 信号质量
        quality_factors.append(self.analysis.signal_quality)
        
        return sum(quality_factors) / len(quality_factors)
    
    def get_actionability_score(self) -> float:
        """获取可执行性评分"""
        actionability = 0.0
        
        # 有明确交易信号
        if self.analysis.trading_signals:
            actionability += 0.4
            
        # 有具体配置建议
        if self.analysis.allocation_recommendation.recommended_allocation:
            actionability += 0.3
            
        # 有风险控制措施
        if self.analysis.risk_assessment.hedging_suggestions:
            actionability += 0.2
            
        # 有监控要点
        if self.analysis.monitoring_points:
            actionability += 0.1
            
        return actionability 