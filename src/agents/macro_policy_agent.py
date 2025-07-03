"""
MacroPolicy Agent - 宏观政策分析Agent
专注于美联储政策、经济数据分析和政策影响评估
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
import asyncio
from dataclasses import dataclass

from .base_agent import BaseAgent
from data_sources.fred_api import FredDataSource
from models.macro_events import (
    EconomicIndicator, FOMCMeeting, FedWatchData, 
    MacroEnvironment, PolicyStance, EconomicIndicatorType
)
from utils.helpers import calculate_percentage_change, moving_average


@dataclass
class PolicyAnalysis:
    """政策分析结果"""
    policy_stance: PolicyStance
    confidence: float
    key_indicators: List[str]
    risk_factors: List[str]
    recommendations: List[str]
    next_meeting_probability: Dict[str, float]
    sentiment_score: float  # -1到1, -1最鸽派, 1最鹰派


class MacroPolicyAgent(BaseAgent):
    """宏观政策分析Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MacroPolicyAgent", config)
        self.fred_source: Optional[FredDataSource] = None
        
        # 关键指标权重
        self.indicator_weights = {
            'FEDFUNDS': 0.25,     # 联邦基金利率
            'UNRATE': 0.20,       # 失业率
            'CPIAUCSL': 0.20,     # CPI
            'GDP': 0.15,          # GDP
            'DGS10': 0.10,        # 10年期国债收益率
            'PAYEMS': 0.10        # 非农就业
        }
        
        # 政策立场判断阈值
        self.policy_thresholds = {
            'hawkish': 0.6,       # 鹰派阈值
            'dovish': -0.6,       # 鸽派阈值
            'neutral': 0.2        # 中性区间
        }
        
    async def initialize(self) -> None:
        """初始化Agent"""
        await super().initialize()
        
        # 初始化FRED数据源
        fred_config = {
            'api_key': self.config.get('fred_api_key'),
            'timeout': 30
        }
        
        self.fred_source = FredDataSource(fred_config)
        await self.fred_source.initialize()
        
        self.logger.info("MacroPolicyAgent初始化完成")
        
    async def fetch_data(self) -> Dict[str, Any]:
        """获取宏观经济数据"""
        if not self.fred_source:
            raise RuntimeError("FRED数据源未初始化")
            
        try:
            # 获取关键经济指标
            indicators = list(self.indicator_weights.keys())
            data = await self.fred_source.fetch_data(
                symbols=indicators,
                start_date=datetime.now() - timedelta(days=730),  # 2年数据
                end_date=datetime.now()
            )
            
            # 获取更多政策相关指标
            additional_indicators = [
                'PCEPILFE',    # 核心PCE
                'INDPRO',      # 工业生产
                'HOUST',       # 新屋开工
                'UMCSENT',     # 密歇根消费者信心
                'WALCL',       # 美联储资产负债表
                'RRPONTSYD'    # 隔夜逆回购
            ]
            
            additional_data = await self.fred_source.fetch_data(
                symbols=additional_indicators,
                start_date=datetime.now() - timedelta(days=365),
                end_date=datetime.now()
            )
            
            # 合并数据
            all_data = data['data'] + additional_data['data']
            
            return {
                'source': 'FRED',
                'timestamp': datetime.now(),
                'indicators': all_data,
                'total_records': len(all_data)
            }
            
        except Exception as e:
            self.logger.error(f"获取宏观数据失败: {e}")
            raise
            
    async def analyze(self, data: Dict[str, Any]) -> PolicyAnalysis:
        """分析宏观政策环境"""
        try:
            indicators = data.get('indicators', [])
            if not indicators:
                raise ValueError("无可用的经济指标数据")
            
            # 按指标类型分组
            indicator_groups = self._group_indicators(indicators)
            
            # 计算各指标的趋势和变化
            trends = await self._calculate_trends(indicator_groups)
            
            # 评估政策立场
            policy_stance = self._evaluate_policy_stance(trends)
            
            # 计算置信度
            confidence = self._calculate_confidence(trends)
            
            # 识别关键指标
            key_indicators = self._identify_key_indicators(trends)
            
            # 评估风险因素
            risk_factors = self._assess_risk_factors(trends)
            
            # 生成建议
            recommendations = self._generate_recommendations(policy_stance, trends)
            
            # 预测下次会议概率
            next_meeting_prob = self._predict_next_meeting(trends)
            
            # 计算情绪分数
            sentiment_score = self._calculate_sentiment_score(trends)
            
            return PolicyAnalysis(
                policy_stance=policy_stance,
                confidence=confidence,
                key_indicators=key_indicators,
                risk_factors=risk_factors,
                recommendations=recommendations,
                next_meeting_probability=next_meeting_prob,
                sentiment_score=sentiment_score
            )
            
        except Exception as e:
            self.logger.error(f"分析宏观政策失败: {e}")
            raise
            
    def _group_indicators(self, indicators: List[EconomicIndicator]) -> Dict[str, List[EconomicIndicator]]:
        """按指标类型分组"""
        groups = {}
        for indicator in indicators:
            symbol = indicator.symbol
            if symbol not in groups:
                groups[symbol] = []
            groups[symbol].append(indicator)
        
        # 按时间排序
        for symbol in groups:
            groups[symbol].sort(key=lambda x: x.release_date)
            
        return groups
        
    async def _calculate_trends(self, indicator_groups: Dict[str, List[EconomicIndicator]]) -> Dict[str, Any]:
        """计算指标趋势"""
        trends = {}
        
        for symbol, indicators in indicator_groups.items():
            if len(indicators) < 2:
                continue
                
            latest = indicators[-1]
            previous = indicators[-2] if len(indicators) >= 2 else None
            
            # 计算变化
            change = None
            change_percent = None
            if previous:
                change = latest.value - previous.value
                change_percent = calculate_percentage_change(previous.value, latest.value)
            
            # 计算移动平均
            values = [ind.value for ind in indicators[-12:]]  # 最近12个数据点
            ma_3 = moving_average(values[-3:]) if len(values) >= 3 else None
            ma_6 = moving_average(values[-6:]) if len(values) >= 6 else None
            ma_12 = moving_average(values) if len(values) >= 12 else None
            
            # 判断趋势方向
            trend_direction = "neutral"
            if ma_3 and ma_6:
                if ma_3 > ma_6 * 1.02:  # 上升趋势
                    trend_direction = "rising"
                elif ma_3 < ma_6 * 0.98:  # 下降趋势
                    trend_direction = "falling"
                    
            trends[symbol] = {
                'latest_value': latest.value,
                'previous_value': previous.value if previous else None,
                'change': change,
                'change_percent': change_percent,
                'ma_3': ma_3,
                'ma_6': ma_6,
                'ma_12': ma_12,
                'trend_direction': trend_direction,
                'indicator_type': latest.indicator_type,
                'latest_date': latest.release_date
            }
            
        return trends
        
    def _evaluate_policy_stance(self, trends: Dict[str, Any]) -> PolicyStance:
        """评估政策立场"""
        stance_score = 0.0
        
        # 根据各指标评估政策偏向
        for symbol, weight in self.indicator_weights.items():
            if symbol not in trends:
                continue
                
            trend = trends[symbol]
            indicator_score = self._get_indicator_policy_score(symbol, trend)
            stance_score += indicator_score * weight
            
        # 根据分数确定政策立场
        if stance_score > self.policy_thresholds['hawkish']:
            return PolicyStance.HAWKISH
        elif stance_score < self.policy_thresholds['dovish']:
            return PolicyStance.DOVISH
        elif abs(stance_score) < self.policy_thresholds['neutral']:
            return PolicyStance.NEUTRAL
        else:
            return PolicyStance.MIXED
            
    def _get_indicator_policy_score(self, symbol: str, trend: Dict[str, Any]) -> float:
        """获取单个指标的政策分数"""
        # 根据不同指标的特性计算政策倾向分数
        # 正数表示鹰派，负数表示鸽派
        
        if symbol == 'FEDFUNDS':  # 联邦基金利率
            # 利率上升=鹰派
            return trend['change_percent'] if trend['change_percent'] else 0
            
        elif symbol == 'UNRATE':  # 失业率
            # 失业率下降=鹰派（经济强劲）
            return -(trend['change_percent'] if trend['change_percent'] else 0)
            
        elif symbol in ['CPIAUCSL', 'PCEPILFE']:  # 通胀指标
            # 通胀上升=鹰派
            return trend['change_percent'] if trend['change_percent'] else 0
            
        elif symbol == 'GDP':  # GDP
            # GDP强劲增长=鹰派
            return trend['change_percent'] if trend['change_percent'] else 0
            
        elif symbol == 'DGS10':  # 10年期国债收益率
            # 收益率上升=鹰派预期
            return trend['change_percent'] if trend['change_percent'] else 0
            
        elif symbol == 'PAYEMS':  # 非农就业
            # 就业增长=鹰派
            return trend['change_percent'] if trend['change_percent'] else 0
            
        return 0.0
        
    def _calculate_confidence(self, trends: Dict[str, Any]) -> float:
        """计算分析置信度"""
        # 基于数据完整性和一致性计算置信度
        available_indicators = len([s for s in self.indicator_weights.keys() if s in trends])
        total_indicators = len(self.indicator_weights)
        
        data_completeness = available_indicators / total_indicators
        
        # 检查趋势一致性
        trend_consistency = self._check_trend_consistency(trends)
        
        # 综合置信度
        confidence = (data_completeness * 0.6 + trend_consistency * 0.4)
        return min(max(confidence, 0.0), 1.0)
        
    def _check_trend_consistency(self, trends: Dict[str, Any]) -> float:
        """检查趋势一致性"""
        # 检查各指标趋势是否一致
        policy_scores = []
        for symbol, weight in self.indicator_weights.items():
            if symbol in trends:
                score = self._get_indicator_policy_score(symbol, trends[symbol])
                policy_scores.append(score)
        
        if not policy_scores:
            return 0.0
            
        # 计算一致性（标准差的倒数）
        import statistics
        if len(policy_scores) > 1:
            stdev = statistics.stdev(policy_scores)
            consistency = 1 / (1 + stdev)  # 标准差越小，一致性越高
        else:
            consistency = 1.0
            
        return consistency
        
    def _identify_key_indicators(self, trends: Dict[str, Any]) -> List[str]:
        """识别关键指标"""
        key_indicators = []
        
        for symbol, trend in trends.items():
            # 判断指标是否关键
            if abs(trend.get('change_percent', 0)) > 5:  # 变化超过5%
                key_indicators.append(f"{symbol}: {trend['trend_direction']}")
                
        return key_indicators
        
    def _assess_risk_factors(self, trends: Dict[str, Any]) -> List[str]:
        """评估风险因素"""
        risk_factors = []
        
        # 通胀风险
        if 'CPIAUCSL' in trends:
            cpi_trend = trends['CPIAUCSL']
            if cpi_trend.get('change_percent', 0) > 3:
                risk_factors.append("通胀压力上升")
                
        # 失业率风险
        if 'UNRATE' in trends:
            unemployment_trend = trends['UNRATE']
            if unemployment_trend.get('change_percent', 0) > 10:
                risk_factors.append("失业率急剧上升")
                
        # 经济衰退风险
        if 'GDP' in trends:
            gdp_trend = trends['GDP']
            if gdp_trend.get('change_percent', 0) < -2:
                risk_factors.append("经济增长放缓")
                
        return risk_factors
        
    def _generate_recommendations(self, policy_stance: PolicyStance, trends: Dict[str, Any]) -> List[str]:
        """生成交易建议"""
        recommendations = []
        
        if policy_stance == PolicyStance.HAWKISH:
            recommendations.extend([
                "预期美联储加息，关注利率敏感板块",
                "考虑做空债券或购买浮动利率产品",
                "关注美元走强对商品价格的影响"
            ])
        elif policy_stance == PolicyStance.DOVISH:
            recommendations.extend([
                "预期美联储降息，关注成长股机会",
                "考虑做多债券或利率敏感资产",
                "关注流动性驱动的资产泡沫风险"
            ])
        elif policy_stance == PolicyStance.NEUTRAL:
            recommendations.extend([
                "政策环境相对稳定，关注基本面驱动",
                "平衡配置，等待更明确的政策信号",
                "关注经济数据变化的边际影响"
            ])
        else:  # MIXED
            recommendations.extend([
                "政策信号混乱，增加风险管理",
                "等待更清晰的政策方向",
                "减少杠杆，提高资金配置灵活性"
            ])
            
        return recommendations
        
    def _predict_next_meeting(self, trends: Dict[str, Any]) -> Dict[str, float]:
        """预测下次FOMC会议概率"""
        # 基于当前趋势预测下次会议决策概率
        stance_score = 0.0
        
        for symbol, weight in self.indicator_weights.items():
            if symbol in trends:
                score = self._get_indicator_policy_score(symbol, trends[symbol])
                stance_score += score * weight
                
        # 转换为概率
        if stance_score > 0.5:
            return {"加息": 0.7, "维持": 0.25, "降息": 0.05}
        elif stance_score < -0.5:
            return {"加息": 0.05, "维持": 0.25, "降息": 0.7}
        else:
            return {"加息": 0.15, "维持": 0.7, "降息": 0.15}
            
    def _calculate_sentiment_score(self, trends: Dict[str, Any]) -> float:
        """计算情绪分数"""
        stance_score = 0.0
        
        for symbol, weight in self.indicator_weights.items():
            if symbol in trends:
                score = self._get_indicator_policy_score(symbol, trends[symbol])
                stance_score += score * weight
                
        # 归一化到-1到1
        return max(min(stance_score / 10, 1.0), -1.0)
        
    async def generate_report(self, analysis: PolicyAnalysis) -> str:
        """生成分析报告"""
        report = f"""
# 宏观政策分析报告

## 📊 政策立场评估
**当前立场**: {analysis.policy_stance.value}
**置信度**: {analysis.confidence:.1%}
**情绪分数**: {analysis.sentiment_score:.2f} (鹰派: +1, 鸽派: -1)

## 🔍 关键指标
{chr(10).join(f"• {indicator}" for indicator in analysis.key_indicators)}

## ⚠️ 风险因素
{chr(10).join(f"• {risk}" for risk in analysis.risk_factors)}

## 💡 交易建议
{chr(10).join(f"• {rec}" for rec in analysis.recommendations)}

## 📈 下次FOMC会议预测
{chr(10).join(f"• {action}: {prob:.1%}" for action, prob in analysis.next_meeting_probability.items())}

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return report
        
    async def get_dependencies(self) -> Set[str]:
        """获取Agent依赖"""
        return {"FredDataSource"}
        
    async def cleanup(self) -> None:
        """清理资源"""
        if self.fred_source:
            await self.fred_source.cleanup()
        # 调用父类清理（如果存在）
        if hasattr(super(), 'cleanup'):
            await super().cleanup() 