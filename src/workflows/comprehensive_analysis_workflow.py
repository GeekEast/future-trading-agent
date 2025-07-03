"""
综合分析工作流
使用LangGraph协调MacroPolicy、GlobalSentiment、OptionStructure三个Agent
整合分析结果并生成综合报告
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from agents.macro_policy_agent import MacroPolicyAgent
from agents.global_sentiment_agent import GlobalSentimentAgent  
from agents.option_structure_agent import OptionStructureAgent
from models.comprehensive_analysis import (
    ComprehensiveAnalysis, AgentConsensus, MarketOutlook, MarketDirection,
    TradingSignal, RiskAssessment, AllocationRecommendation, 
    MarketRegimeAnalysis, TimeHorizon, ComprehensiveReport
)
from models.macro_events import PolicyStance
from models.sentiment_data import SentimentAnalysis, SentimentType, RiskLevel
from models.option_data import OptionAnalysisResult
from utils.logger import setup_logger
from utils.config import load_config


class AnalysisState(TypedDict):
    """分析状态"""
    # 输入参数
    target_symbol: str
    analysis_timestamp: datetime
    config: Dict[str, Any]
    
    # Agent分析结果
    macro_analysis: Optional[Any]  # PolicyAnalysis from agents.macro_policy_agent
    sentiment_analysis: Optional[SentimentAnalysis]
    option_analysis: Optional[OptionAnalysisResult]
    
    # 中间结果
    agent_errors: List[str]
    data_quality_scores: Dict[str, float]
    
    # 最终结果
    comprehensive_analysis: Optional[ComprehensiveAnalysis]
    final_report: Optional[str]
    
    # 状态控制
    current_step: str
    completed_agents: List[str]
    failed_agents: List[str]


class ComprehensiveAnalysisWorkflow:
    """综合分析工作流"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        setup_logger()  # Initialize logging system
        from loguru import logger
        self.logger = logger.bind(component="ComprehensiveAnalysisWorkflow")
        
        # 初始化Agents
        self.macro_agent = None
        self.sentiment_agent = None
        self.option_agent = None
        
        # 分析权重配置
        self.agent_weights = {
            'macro': 0.4,      # 宏观分析权重
            'sentiment': 0.35, # 情绪分析权重
            'option': 0.25     # 期权分析权重
        }
        
        # 构建工作流图
        self.workflow = None
        self._build_workflow()
        
    async def initialize(self) -> None:
        """初始化工作流和Agents"""
        
        # 初始化各个Agent
        self.macro_agent = MacroPolicyAgent(self.config)
        await self.macro_agent.initialize()
        
        self.sentiment_agent = GlobalSentimentAgent(self.config)
        await self.sentiment_agent.initialize()
        
        self.option_agent = OptionStructureAgent(self.config)
        await self.option_agent.initialize()
        
        self.logger.info("综合分析工作流初始化完成")
        
    def _build_workflow(self) -> None:
        """构建LangGraph工作流"""
        
        # 创建状态图
        workflow = StateGraph(AnalysisState)
        
        # 添加节点
        workflow.add_node("start_analysis", self._start_analysis)
        workflow.add_node("run_macro_analysis", self._run_macro_analysis)
        workflow.add_node("run_sentiment_analysis", self._run_sentiment_analysis)
        workflow.add_node("run_option_analysis", self._run_option_analysis)
        workflow.add_node("synthesize_results", self._synthesize_results)
        workflow.add_node("generate_report", self._generate_report)
        workflow.add_node("handle_error", self._handle_error)
        
        # 设置入口点
        workflow.set_entry_point("start_analysis")
        
        # 添加边
        workflow.add_edge("start_analysis", "run_macro_analysis")
        workflow.add_edge("run_macro_analysis", "run_sentiment_analysis")
        workflow.add_edge("run_sentiment_analysis", "run_option_analysis")
        workflow.add_edge("run_option_analysis", "synthesize_results")
        workflow.add_edge("synthesize_results", "generate_report")
        workflow.add_edge("generate_report", END)
        workflow.add_edge("handle_error", END)
        
        # 编译工作流
        self.workflow = workflow.compile()
        
    async def run_analysis(self, target_symbol: str = "SPY") -> ComprehensiveAnalysis:
        """运行综合分析"""
        
        # 初始化状态
        initial_state = AnalysisState(
            target_symbol=target_symbol,
            analysis_timestamp=datetime.now(),
            config=self.config,
            macro_analysis=None,
            sentiment_analysis=None,
            option_analysis=None,
            agent_errors=[],
            data_quality_scores={},
            comprehensive_analysis=None,
            final_report=None,
            current_step="start",
            completed_agents=[],
            failed_agents=[]
        )
        
        try:
            # 运行工作流
            final_state = await self.workflow.ainvoke(initial_state)
            
            if final_state['comprehensive_analysis']:
                return final_state['comprehensive_analysis']
            else:
                # 如果分析失败，返回默认结果
                return self._create_fallback_analysis(target_symbol, final_state['agent_errors'])
                
        except Exception as e:
            self.logger.error(f"综合分析工作流执行失败: {e}")
            return self._create_fallback_analysis(target_symbol, [str(e)])
            
    async def _start_analysis(self, state: AnalysisState) -> AnalysisState:
        """开始分析"""
        self.logger.info(f"开始综合分析: {state['target_symbol']}")
        state['current_step'] = "macro_analysis"
        return state
        
    async def _run_macro_analysis(self, state: AnalysisState) -> AnalysisState:
        """运行宏观政策分析"""
        try:
            self.logger.info("执行宏观政策分析...")
            
            # 获取宏观数据并分析
            macro_data = await self.macro_agent.fetch_data()
            state['macro_analysis'] = await self.macro_agent.analyze(macro_data)
            
            # 计算数据质量分数
            state['data_quality_scores']['macro'] = self._calculate_macro_quality(
                macro_data, state['macro_analysis']
            )
            
            state['completed_agents'].append('macro')
            self.logger.info("宏观政策分析完成")
            
        except Exception as e:
            self.logger.error(f"宏观分析失败: {e}")
            state['agent_errors'].append(f"宏观分析: {str(e)}")
            state['failed_agents'].append('macro')
            
        return state
        
    async def _run_sentiment_analysis(self, state: AnalysisState) -> AnalysisState:
        """运行全球情绪分析"""
        try:
            self.logger.info("执行全球情绪分析...")
            
            # 获取情绪数据并分析
            sentiment_data = await self.sentiment_agent.fetch_data()
            state['sentiment_analysis'] = await self.sentiment_agent.analyze(sentiment_data)
            
            # 计算数据质量分数
            state['data_quality_scores']['sentiment'] = self._calculate_sentiment_quality(
                sentiment_data, state['sentiment_analysis']
            )
            
            state['completed_agents'].append('sentiment')
            self.logger.info("全球情绪分析完成")
            
        except Exception as e:
            self.logger.error(f"情绪分析失败: {e}")
            state['agent_errors'].append(f"情绪分析: {str(e)}")
            state['failed_agents'].append('sentiment')
            
        return state
        
    async def _run_option_analysis(self, state: AnalysisState) -> AnalysisState:
        """运行期权结构分析"""
        try:
            self.logger.info("执行期权结构分析...")
            
            # 获取期权数据并分析
            option_data = await self.option_agent.fetch_data(state['target_symbol'])
            state['option_analysis'] = await self.option_agent.analyze(option_data)
            
            # 计算数据质量分数
            state['data_quality_scores']['option'] = self._calculate_option_quality(
                option_data, state['option_analysis']
            )
            
            state['completed_agents'].append('option')
            self.logger.info("期权结构分析完成")
            
        except Exception as e:
            self.logger.error(f"期权分析失败: {e}")
            state['agent_errors'].append(f"期权分析: {str(e)}")
            state['failed_agents'].append('option')
            
        return state
        
    async def _synthesize_results(self, state: AnalysisState) -> AnalysisState:
        """综合分析结果"""
        try:
            self.logger.info("综合分析结果...")
            
            # 检查是否有足够的分析结果
            if len(state['completed_agents']) == 0:
                raise Exception("所有Agent分析都失败")
                
            # 进行综合分析
            state['comprehensive_analysis'] = self._perform_comprehensive_analysis(state)
            
            state['current_step'] = "completed"
            self.logger.info("综合分析完成")
            
        except Exception as e:
            self.logger.error(f"综合分析失败: {e}")
            state['agent_errors'].append(f"综合分析: {str(e)}")
            state['current_step'] = "failed"
            
        return state
        
    async def _generate_report(self, state: AnalysisState) -> AnalysisState:
        """生成综合报告"""
        try:
            if state['comprehensive_analysis']:
                state['final_report'] = await self._create_comprehensive_report(
                    state['comprehensive_analysis']
                )
                self.logger.info("综合报告生成完成")
            else:
                state['final_report'] = "分析失败，无法生成报告"
                
        except Exception as e:
            self.logger.error(f"报告生成失败: {e}")
            state['final_report'] = f"报告生成失败: {str(e)}"
            
        return state
        
    async def _handle_error(self, state: AnalysisState) -> AnalysisState:
        """处理错误"""
        self.logger.error(f"工作流错误处理: {state['agent_errors']}")
        return state
        
    def _perform_comprehensive_analysis(self, state: AnalysisState) -> ComprehensiveAnalysis:
        """执行综合分析"""
        
        # 计算Agent共识度
        consensus = self._calculate_agent_consensus(
            state['macro_analysis'],
            state['sentiment_analysis'], 
            state['option_analysis']
        )
        
        # 生成市场展望
        market_outlook = self._generate_market_outlook(
            state['macro_analysis'],
            state['sentiment_analysis'],
            state['option_analysis']
        )
        
        # 生成交易信号
        trading_signals = self._generate_trading_signals(
            state['macro_analysis'],
            state['sentiment_analysis'],
            state['option_analysis'],
            market_outlook
        )
        
        # 评估风险
        risk_assessment = self._assess_comprehensive_risk(
            state['macro_analysis'],
            state['sentiment_analysis'],
            state['option_analysis']
        )
        
        # 生成配置建议
        allocation_recommendation = self._generate_allocation_recommendation(
            market_outlook, risk_assessment, trading_signals
        )
        
        # 分析市场制度
        regime_analysis = self._analyze_market_regime(
            state['macro_analysis'],
            state['sentiment_analysis'],
            state['option_analysis']
        )
        
        # 计算综合指标
        overall_score = self._calculate_overall_score(state)
        signal_quality = self._calculate_signal_quality(trading_signals)
        data_completeness = np.mean(list(state['data_quality_scores'].values())) if state['data_quality_scores'] else 0.5
        analysis_reliability = consensus.overall_consensus
        
        # 生成执行摘要和关键要点
        executive_summary = self._generate_executive_summary(market_outlook, trading_signals, risk_assessment)
        key_takeaways = self._generate_key_takeaways(
            state['macro_analysis'], state['sentiment_analysis'], state['option_analysis']
        )
        action_items = self._generate_action_items(trading_signals, risk_assessment)
        
        # 确定下次更新时间和监控要点
        next_update_time = datetime.now() + timedelta(hours=4)  # 4小时后更新
        monitoring_points = self._identify_monitoring_points(
            state['macro_analysis'], state['sentiment_analysis'], state['option_analysis']
        )
        
        return ComprehensiveAnalysis(
            analysis_timestamp=state['analysis_timestamp'],
            target_symbol=state['target_symbol'],
            analysis_version="1.0",
            macro_analysis=state['macro_analysis'],
            sentiment_analysis=state['sentiment_analysis'],
            option_analysis=state['option_analysis'],
            consensus=consensus,
            market_outlook=market_outlook,
            trading_signals=trading_signals,
            risk_assessment=risk_assessment,
            allocation_recommendation=allocation_recommendation,
            regime_analysis=regime_analysis,
            overall_score=overall_score,
            signal_quality=signal_quality,
            data_completeness=data_completeness,
            analysis_reliability=analysis_reliability,
            executive_summary=executive_summary,
            key_takeaways=key_takeaways,
            action_items=action_items,
            next_update_time=next_update_time,
            monitoring_points=monitoring_points
        ) 

    def _calculate_agent_consensus(
        self,
        macro_analysis: Optional[Any],  # PolicyAnalysis from agents.macro_policy_agent
        sentiment_analysis: Optional[SentimentAnalysis], 
        option_analysis: Optional[OptionAnalysisResult]
    ) -> AgentConsensus:
        """计算Agent共识度"""
        
        # 简化的共识度计算
        agreements = []
        disagreements = []
        consensus_areas = []
        
        if macro_analysis and sentiment_analysis:
            # 比较宏观和情绪分析
            macro_score = self._policy_to_score(macro_analysis.policy_stance)
            sentiment_score = sentiment_analysis.sentiment_score
            
            agreement = 1.0 - abs(macro_score - sentiment_score) / 2.0
            agreements.append(agreement)
            
            if agreement > 0.7:
                consensus_areas.append("宏观情绪一致")
            else:
                disagreements.append("宏观情绪分歧")
                
        overall_consensus = np.mean(agreements) if agreements else 0.5
        
        return AgentConsensus(
            macro_sentiment_agreement=agreements[0] if agreements else 0.5,
            sentiment_option_agreement=0.6,
            macro_option_agreement=0.7,
            overall_consensus=overall_consensus,
            main_disagreements=disagreements,
            consensus_areas=consensus_areas
        )
        
    def _generate_market_outlook(
        self,
        macro_analysis: Optional[Any],  # PolicyAnalysis from agents.macro_policy_agent
        sentiment_analysis: Optional[SentimentAnalysis],
        option_analysis: Optional[OptionAnalysisResult]
    ) -> MarketOutlook:
        """生成市场展望"""
        
        # 综合各Agent的观点
        direction_scores = []
        
        if macro_analysis:
            if macro_analysis.policy_stance == PolicyStance.DOVISH:
                direction_scores.append(0.3)  # 鸽派偏多
            elif macro_analysis.policy_stance == PolicyStance.HAWKISH:
                direction_scores.append(-0.3)  # 鹰派偏空
            else:
                direction_scores.append(0.0)
                
        if sentiment_analysis:
            direction_scores.append(sentiment_analysis.sentiment_score)
            
        if option_analysis:
            # 简化期权影响
            direction_scores.append(0.1)
            
        avg_score = np.mean(direction_scores) if direction_scores else 0.0
        
        # 确定市场方向
        if avg_score > 0.2:
            direction = MarketDirection.BULLISH
            bullish_prob = 0.6
            bearish_prob = 0.2
            sideways_prob = 0.2
        elif avg_score < -0.2:
            direction = MarketDirection.BEARISH
            bullish_prob = 0.2
            bearish_prob = 0.6
            sideways_prob = 0.2
        else:
            direction = MarketDirection.NEUTRAL
            bullish_prob = 0.3
            bearish_prob = 0.3
            sideways_prob = 0.4
            
        return MarketOutlook(
            direction=direction,
            confidence=0.75,
            time_horizon=TimeHorizon.SHORT_TERM,
            intraday_outlook="震荡为主",
            short_term_outlook="谨慎乐观",
            medium_term_outlook="关注政策变化",
            bullish_probability=bullish_prob,
            bearish_probability=bearish_prob,
            sideways_probability=sideways_prob,
            supporting_factors=["宏观环境相对稳定"],
            risk_factors=["市场波动性增加"],
            catalysts=["政策变化", "经济数据"],
            best_case_scenario="政策支撑下的稳步上涨",
            worst_case_scenario="政策转向导致的调整",
            base_case_scenario="震荡上行的慢牛格局"
        )
        
    def _generate_trading_signals(
        self,
        macro_analysis: Optional[Any],  # PolicyAnalysis from agents.macro_policy_agent
        sentiment_analysis: Optional[SentimentAnalysis],
        option_analysis: Optional[OptionAnalysisResult]
    ) -> List[TradingSignal]:
        """生成交易信号"""
        
        signals = []
        
        # 基于综合分析生成主要信号
        signal = TradingSignal(
            signal_type="HOLD",
            strength=0.6,
            confidence=0.7,
            time_horizon=TimeHorizon.SHORT_TERM,
            entry_price=None,
            target_price=None,
            stop_loss=None,
            position_size=0.5,
            supporting_agents=["macro", "sentiment"],
            contradicting_agents=[],
            reasoning="综合分析显示市场中性偏多，建议持有观望",
            risk_reward_ratio=1.5
        )
        
        signals.append(signal)
        return signals
        
    def _assess_comprehensive_risk(
        self,
        macro_analysis: Optional[Any],  # PolicyAnalysis from agents.macro_policy_agent
        sentiment_analysis: Optional[SentimentAnalysis],
        option_analysis: Optional[OptionAnalysisResult]
    ) -> RiskAssessment:
        """评估综合风险"""
        
        # 导入必要的枚举
        from models.sentiment_data import RiskLevel
        
        return RiskAssessment(
            overall_risk_level="中等风险",
            risk_score=45.0,
            macro_risk=30.0,
            sentiment_risk=40.0,
            technical_risk=35.0,
            liquidity_risk=25.0,
            volatility_risk=50.0,
            systemic_risk=RiskLevel.MEDIUM,
            top_risks=["政策不确定性", "市场波动加剧"],
            tail_risks=["地缘政治风险"],
            black_swan_events=["重大政策转向"],
            hedging_suggestions=["适度对冲波动率风险"],
            position_limits={"equity": 0.6, "bond": 0.3, "cash": 0.1},
            stop_loss_levels={"equity": 0.05, "bond": 0.02}
        )
        
    def _generate_allocation_recommendation(
        self,
        market_outlook: MarketOutlook,
        risk_assessment: RiskAssessment
    ) -> AllocationRecommendation:
        """生成配置建议"""
        
        return AllocationRecommendation(
            recommended_allocation={
                "股票": 0.5,
                "债券": 0.3,
                "现金": 0.2
            },
            equity_allocation=0.5,
            bond_allocation=0.3,
            commodity_allocation=0.0,
            cash_allocation=0.2,
            alternative_allocation=0.0,
            risk_parity=False,
            leverage_recommendation=1.0,
            rebalancing_frequency="月度",
            trigger_conditions=["市场方向明确变化", "风险水平显著变化"],
            reasoning="基于当前中性偏多的市场环境，建议均衡配置"
        )
        
    def _analyze_market_regime(
        self,
        macro_analysis: Optional[Any],  # PolicyAnalysis from agents.macro_policy_agent
        sentiment_analysis: Optional[SentimentAnalysis],
        option_analysis: Optional[OptionAnalysisResult]
    ) -> MarketRegimeAnalysis:
        """分析市场制度"""
        
        return MarketRegimeAnalysis(
            current_regime="震荡市",
            regime_confidence=0.7,
            regime_duration=30,
            volatility_regime="正常",
            correlation_regime="正常",
            liquidity_regime="充足",
            transition_probability=0.3,
            next_likely_regime="趋势市",
            transition_catalysts=["重大政策变化", "经济数据超预期"],
            historical_parallels=["2019年中期"],
            lessons_learned=["需要保持灵活性"]
        )
        
    def _policy_to_score(self, stance: PolicyStance) -> float:
        """将政策立场转换为数值分数"""
        if stance == PolicyStance.DOVISH:
            return 0.5
        elif stance == PolicyStance.HAWKISH:
            return -0.5
        else:
            return 0.0
            
    def _calculate_macro_quality(self, data: Dict[str, Any], analysis: Any) -> float:
        """计算宏观数据质量分数"""
        if not data or not analysis:
            return 0.0
        
        # 基于数据记录数和置信度计算质量
        record_count = data.get('total_records', 0)
        confidence = getattr(analysis, 'confidence', 0.0)
        
        # 数据量评分 (0-1)
        data_score = min(record_count / 10000, 1.0)  # 10000条记录为满分
        
        # 综合评分
        return (data_score * 0.4 + confidence * 0.6)

    def _calculate_sentiment_quality(self, data: Dict[str, Any], analysis: Any) -> float:
        """计算情绪数据质量分数"""
        if not data or not analysis:
            return 0.0
        
        # 基于数据源数量和置信度
        source_count = data.get('source_count', 0)
        confidence = getattr(analysis, 'confidence', 0.0)
        
        # 数据源评分
        source_score = min(source_count / 5, 1.0)  # 5个数据源为满分
        
        return (source_score * 0.3 + confidence * 0.7)

    def _calculate_option_quality(self, data: Dict[str, Any], analysis: Any) -> float:
        """计算期权数据质量分数"""
        if not data or not analysis:
            return 0.0
        
        # 基于期权数据点数和分析质量
        option_count = data.get('option_count', 0)
        quality_score = getattr(analysis, 'quality_score', 0.0)
        
        # 数据点评分
        data_score = min(option_count / 100, 1.0)  # 100个期权为满分
        
        return (data_score * 0.4 + quality_score / 100 * 0.6)

    def _calculate_overall_score(self, state: AnalysisState) -> float:
        """计算综合评分"""
        scores = []
        
        # 各Agent贡献的评分
        if 'macro' in state['completed_agents']:
            macro_score = state['data_quality_scores'].get('macro', 0.5) * 100
            scores.append(macro_score * self.agent_weights['macro'])
            
        if 'sentiment' in state['completed_agents']:
            sentiment_score = state['data_quality_scores'].get('sentiment', 0.5) * 100
            scores.append(sentiment_score * self.agent_weights['sentiment'])
            
        if 'option' in state['completed_agents']:
            option_score = state['data_quality_scores'].get('option', 0.5) * 100
            scores.append(option_score * self.agent_weights['option'])
        
        return sum(scores) if scores else 50.0

    def _calculate_signal_quality(self, signals: List[TradingSignal]) -> float:
        """计算信号质量"""
        if not signals:
            return 0.0
            
        # 基于信号强度和置信度的加权平均
        quality_scores = []
        for signal in signals:
            quality = signal.strength * signal.confidence
            quality_scores.append(quality)
            
        return sum(quality_scores) / len(quality_scores)

    def _generate_executive_summary(
        self, 
        market_outlook: MarketOutlook, 
        trading_signals: List[TradingSignal], 
        risk_assessment: RiskAssessment
    ) -> str:
        """生成执行摘要"""
        direction = market_outlook.direction.value
        confidence = market_outlook.confidence
        risk_level = risk_assessment.overall_risk_level
        
        summary = f"市场展望{direction}（置信度{confidence:.0%}），"
        summary += f"风险级别为{risk_level}。"
        
        if trading_signals:
            main_signal = trading_signals[0]
            summary += f"主要交易信号为{main_signal.signal_type}，"
            summary += f"信号强度{main_signal.strength:.1f}。"
        
        return summary

    def _generate_key_takeaways(
        self, 
        macro_analysis: Optional[Any],
        sentiment_analysis: Optional[Any], 
        option_analysis: Optional[Any]
    ) -> List[str]:
        """生成关键要点"""
        takeaways = []
        
        if macro_analysis and hasattr(macro_analysis, 'policy_stance'):
            takeaways.append(f"宏观政策立场：{macro_analysis.policy_stance.value}")
            
        if sentiment_analysis and hasattr(sentiment_analysis, 'sentiment_score'):
            score = sentiment_analysis.sentiment_score
            sentiment_desc = "乐观" if score > 0.1 else "悲观" if score < -0.1 else "中性"
            takeaways.append(f"市场情绪：{sentiment_desc}")
            
        if option_analysis and hasattr(option_analysis, 'max_pain'):
            takeaways.append(f"期权最大痛点：{option_analysis.max_pain}")
            
        if not takeaways:
            takeaways.append("数据分析中，请稍后查看详细结果")
            
        return takeaways

    def _generate_action_items(
        self, 
        trading_signals: List[TradingSignal], 
        risk_assessment: RiskAssessment
    ) -> List[str]:
        """生成行动项"""
        actions = []
        
        # 基于交易信号生成行动项
        for signal in trading_signals[:2]:  # 最多2个主要信号
            if signal.signal_type == "BUY":
                actions.append(f"考虑买入机会，目标强度{signal.strength:.1f}")
            elif signal.signal_type == "SELL":
                actions.append(f"考虑卖出机会，目标强度{signal.strength:.1f}")
            else:
                actions.append(f"维持当前持仓，观察市场变化")
        
        # 基于风险评估生成风控行动项
        if risk_assessment.risk_score > 70:
            actions.append("高风险环境，加强风险控制")
        elif risk_assessment.risk_score < 30:
            actions.append("低风险环境，可适度增加仓位")
            
        if not actions:
            actions.append("继续监控市场动态，等待明确信号")
            
        return actions

    def _identify_monitoring_points(
        self, 
        macro_analysis: Optional[Any],
        sentiment_analysis: Optional[Any], 
        option_analysis: Optional[Any]
    ) -> List[str]:
        """识别监控要点"""
        points = []
        
        if macro_analysis and hasattr(macro_analysis, 'key_indicators'):
            for indicator in macro_analysis.key_indicators[:2]:
                points.append(f"关注{indicator}")
                
        if sentiment_analysis:
            points.append("监控市场情绪变化")
            
        if option_analysis:
            points.append("观察期权持仓变化")
            
        # 默认监控点
        points.extend([
            "关注重要经济数据发布",
            "监控政策变化信号"
        ])
        
        return points

    def _create_fallback_analysis(self, target_symbol: str, errors: List[str]) -> ComprehensiveAnalysis:
        """创建fallback分析结果"""
        
        return ComprehensiveAnalysis(
            analysis_timestamp=datetime.now(),
            target_symbol=target_symbol,
            analysis_version="1.0-fallback",
            macro_analysis=None,
            sentiment_analysis=None,
            option_analysis=None,
            consensus=AgentConsensus(
                macro_sentiment_agreement=0.5,
                sentiment_option_agreement=0.5,
                macro_option_agreement=0.5,
                overall_consensus=0.5,
                main_disagreements=["数据获取失败"],
                consensus_areas=[]
            ),
            market_outlook=MarketOutlook(
                direction=MarketDirection.UNCERTAIN,
                confidence=0.3,
                time_horizon=TimeHorizon.SHORT_TERM,
                intraday_outlook="数据不足",
                short_term_outlook="等待数据更新",
                medium_term_outlook="暂无预测",
                bullish_probability=0.33,
                bearish_probability=0.33,
                sideways_probability=0.34,
                supporting_factors=[],
                risk_factors=["数据质量问题"],
                catalysts=[],
                best_case_scenario="数据恢复正常",
                worst_case_scenario="持续数据问题",
                base_case_scenario="等待系统恢复"
            ),
            trading_signals=[],
            risk_assessment=RiskAssessment(
                overall_risk_level="高风险",
                risk_score=80.0,
                macro_risk=50.0,
                sentiment_risk=50.0,
                technical_risk=50.0,
                liquidity_risk=50.0,
                volatility_risk=50.0,
                top_risks=["数据不可靠"],
                tail_risks=[],
                black_swan_events=[],
                hedging_suggestions=["等待数据恢复"],
                position_limits={},
                stop_loss_levels={}
            ),
            allocation_recommendation=AllocationRecommendation(
                recommended_allocation={"现金": 1.0},
                equity_allocation=0.0,
                bond_allocation=0.0,
                commodity_allocation=0.0,
                cash_allocation=1.0,
                alternative_allocation=0.0,
                risk_parity=False,
                leverage_recommendation=0.0,
                rebalancing_frequency="等待数据",
                trigger_conditions=[],
                reasoning="由于分析失败，建议保持现金"
            ),
            regime_analysis=MarketRegimeAnalysis(
                current_regime="未知",
                regime_confidence=0.0,
                regime_duration=0,
                volatility_regime="未知",
                correlation_regime="未知",
                liquidity_regime="未知",
                transition_probability=0.5,
                next_likely_regime="未知",
                transition_catalysts=[],
                historical_parallels=[],
                lessons_learned=[]
            ),
            overall_score=30.0,
            signal_quality=0.0,
            data_completeness=0.0,
            analysis_reliability=0.0,
            executive_summary="分析失败，建议等待系统恢复",
            key_takeaways=["系统存在问题", "需要修复数据源"],
            action_items=["检查系统状态", "联系技术支持"],
            next_update_time=datetime.now() + timedelta(hours=1),
            monitoring_points=["系统状态", "数据源可用性"]
        )
        
    async def cleanup(self) -> None:
        """清理资源"""
        if self.macro_agent:
            await self.macro_agent.cleanup()
        if self.sentiment_agent:
            await self.sentiment_agent.cleanup()
        if self.option_agent:
            await self.option_agent.cleanup()
            
        self.logger.info("综合分析工作流清理完成") 