"""
全球市场情绪分析Agent
分析全球市场情绪、VIX指标、跨资产相关性、地缘政治风险
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from scipy.stats import pearsonr
from collections import defaultdict

from .base_agent import BaseAgent
from data_sources.yahoo_finance import YahooFinanceDataSource
from data_sources.news_source import NewsDataSource
from models.sentiment_data import (
    SentimentAnalysis, SentimentType, RiskLevel, 
    GlobalIndex, VolatilityIndicator, CrossAssetCorrelation,
    MarketRegimeIndicator, FlightToQualitySignal, 
    TechnicalSentiment, SentimentReport, AssetClass
)
from utils.helpers import moving_average, calculate_rsi, calculate_macd


class GlobalSentimentAgent(BaseAgent):
    """全球市场情绪分析Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("GlobalSentiment", config)
        
        # 数据源配置
        self.yahoo_source = None
        self.news_source = None
        
        # 全球主要指数配置
        self.global_indices = {
            # 美国主要指数
            '^GSPC': {'name': 'S&P 500', 'region': 'US', 'weight': 0.30},
            '^DJI': {'name': 'Dow Jones', 'region': 'US', 'weight': 0.20},
            '^IXIC': {'name': 'NASDAQ', 'region': 'US', 'weight': 0.25},
            
            # 国际主要指数
            '^FTSE': {'name': 'FTSE 100', 'region': 'UK', 'weight': 0.08},
            '^GDAXI': {'name': 'DAX', 'region': 'DE', 'weight': 0.08},
            '^N225': {'name': 'Nikkei 225', 'region': 'JP', 'weight': 0.09},
        }
        
        # 波动率指数
        self.volatility_indices = ['^VIX', '^VVIX', '^VSTOXX']
        
        # 避险资产
        self.safe_haven_assets = ['GLD', '^TNX', 'DX-Y.NYB']
        
        # 风险资产
        self.risk_assets = ['QQQ', 'IWM', 'EEM', 'HYG']
        
        # 情绪权重配置
        self.sentiment_weights = {
            'volatility': 0.30,      # 波动率指标
            'correlations': 0.25,    # 跨资产相关性
            'news_sentiment': 0.20,  # 新闻情绪
            'technical': 0.15,       # 技术指标
            'flight_to_quality': 0.10 # 避险流动
        }
        
    async def initialize(self) -> None:
        """初始化GlobalSentiment Agent"""
        
        # 初始化数据源
        self.yahoo_source = YahooFinanceDataSource(self.config)
        await self.yahoo_source.initialize()
        
        self.news_source = NewsDataSource(self.config)
        await self.news_source.initialize()
        
        self.logger.info("GlobalSentiment Agent初始化完成")
        
    async def fetch_data(self) -> Dict[str, Any]:
        """获取全球市场情绪分析所需数据"""
        
        try:
            # 并行获取各类数据
            tasks = [
                self._fetch_global_indices_data(),
                self._fetch_volatility_data(),
                self._fetch_safe_haven_data(),
                self._fetch_risk_assets_data(),
                self._fetch_news_sentiment_data()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            global_indices, volatility_data, safe_haven_data, risk_assets_data, news_data = results
            
            return {
                'timestamp': datetime.now(),
                'global_indices': global_indices if not isinstance(global_indices, Exception) else [],
                'volatility_data': volatility_data if not isinstance(volatility_data, Exception) else [],
                'safe_haven_data': safe_haven_data if not isinstance(safe_haven_data, Exception) else [],
                'risk_assets_data': risk_assets_data if not isinstance(risk_assets_data, Exception) else [],
                'news_data': news_data if not isinstance(news_data, Exception) else {'news_sentiment': []},
                'total_data_points': len(global_indices or []) + len(volatility_data or [])
            }
            
        except Exception as e:
            self.logger.error(f"获取全球情绪数据失败: {e}")
            return {'timestamp': datetime.now(), 'error': str(e)}
            
    async def _fetch_global_indices_data(self) -> List[GlobalIndex]:
        """获取全球主要指数数据"""
        try:
            symbols = list(self.global_indices.keys())
            data = await self.yahoo_source.fetch_data(symbols, data_type='quote')
            
            indices = []
            for data_point in data.get('data', []):
                symbol = data_point.symbol
                if symbol in self.global_indices:
                    config = self.global_indices[symbol]
                    
                    index = GlobalIndex(
                        symbol=symbol,
                        name=config['name'],
                        value=data_point.price,
                        change=data_point.change or 0,
                        change_percent=data_point.change_percent or 0,
                        volume=data_point.volume,
                        market_cap=None,
                        region=config['region'],
                        sector=None,
                        timestamp=data_point.timestamp
                    )
                    indices.append(index)
                    
            return indices
            
        except Exception as e:
            self.logger.error(f"获取全球指数数据失败: {e}")
            return []
            
    async def _fetch_volatility_data(self) -> List[VolatilityIndicator]:
        """获取波动率指标数据"""
        try:
            data = await self.yahoo_source.fetch_data(self.volatility_indices, data_type='quote')
            
            indicators = []
            for data_point in data.get('data', []):
                # 获取历史数据计算百分位
                hist_data = await self.yahoo_source.fetch_data(
                    [data_point.symbol], 
                    data_type='history', 
                    period='1y'
                )
                
                percentile_rank = self._calculate_percentile_rank(
                    data_point.price, hist_data.get('data', [])
                )
                
                interpretation = self._interpret_volatility(data_point.price, percentile_rank)
                
                indicator = VolatilityIndicator(
                    symbol=data_point.symbol,
                    name=self._get_volatility_name(data_point.symbol),
                    value=data_point.price,
                    change=data_point.change or 0,
                    change_percent=data_point.change_percent or 0,
                    percentile_rank=percentile_rank,
                    interpretation=interpretation,
                    timestamp=data_point.timestamp
                )
                indicators.append(indicator)
                
            return indicators
            
        except Exception as e:
            self.logger.error(f"获取波动率数据失败: {e}")
            return []
            
    async def _fetch_safe_haven_data(self) -> List[Dict[str, Any]]:
        """获取避险资产数据"""
        try:
            data = await self.yahoo_source.fetch_data(self.safe_haven_assets, data_type='quote')
            return data.get('data', [])
        except Exception as e:
            self.logger.error(f"获取避险资产数据失败: {e}")
            return []
            
    async def _fetch_risk_assets_data(self) -> List[Dict[str, Any]]:
        """获取风险资产数据"""
        try:
            data = await self.yahoo_source.fetch_data(self.risk_assets, data_type='quote')
            return data.get('data', [])
        except Exception as e:
            self.logger.error(f"获取风险资产数据失败: {e}")
            return []
            
    async def _fetch_news_sentiment_data(self) -> Dict[str, Any]:
        """获取新闻情绪数据"""
        try:
            return await self.news_source.fetch_data(hours_back=24, max_articles=50)
        except Exception as e:
            self.logger.error(f"获取新闻数据失败: {e}")
            return {'news_sentiment': []}
            
    async def analyze(self, data: Dict[str, Any]) -> SentimentAnalysis:
        """分析全球市场情绪"""
        
        try:
            # 分析各个组件
            volatility_sentiment = self._analyze_volatility_sentiment(data.get('volatility_data', []))
            correlation_sentiment = await self._analyze_correlation_sentiment(data)
            news_sentiment = self._analyze_news_sentiment(data.get('news_data', {}))
            technical_sentiment = await self._analyze_technical_sentiment(data)
            flight_to_quality = self._analyze_flight_to_quality(data)
            
            # 综合计算情绪分数
            sentiment_components = {
                'volatility': volatility_sentiment,
                'correlations': correlation_sentiment,
                'news_sentiment': news_sentiment,
                'technical': technical_sentiment,
                'flight_to_quality': flight_to_quality
            }
            
            overall_sentiment_score = self._calculate_weighted_sentiment(sentiment_components)
            overall_sentiment_type = self._determine_sentiment_type(overall_sentiment_score)
            risk_level = self._assess_risk_level(sentiment_components)
            
            # 计算分类情绪
            asset_sentiments = self._calculate_asset_class_sentiments(data)
            
            # 生成关键指标
            fear_greed_index = self._calculate_fear_greed_index(sentiment_components)
            volatility_regime = self._determine_volatility_regime(data.get('volatility_data', []))
            correlation_regime = self._determine_correlation_regime(correlation_sentiment)
            
            # 识别驱动因素和风险
            primary_drivers = self._identify_primary_drivers(sentiment_components, data)
            risk_factors = self._identify_risk_factors(sentiment_components, data)
            opportunities = self._identify_opportunities(sentiment_components, data)
            
            # 生成预测和建议
            short_term_outlook = self._generate_short_term_outlook(sentiment_components)
            medium_term_outlook = self._generate_medium_term_outlook(sentiment_components)
            positioning = self._recommend_positioning(sentiment_components, risk_level)
            hedges = self._suggest_hedges(risk_level, sentiment_components)
            
            # 计算置信度
            confidence = self._calculate_confidence(sentiment_components, data)
            
            return SentimentAnalysis(
                overall_sentiment=overall_sentiment_type,
                sentiment_score=overall_sentiment_score,
                confidence=confidence,
                risk_level=risk_level,
                
                # 分类情绪
                equity_sentiment=asset_sentiments['equity'],
                bond_sentiment=asset_sentiments['bond'],
                commodity_sentiment=asset_sentiments['commodity'],
                currency_sentiment=asset_sentiments['currency'],
                
                # 关键指标
                fear_greed_index=fear_greed_index,
                volatility_regime=volatility_regime,
                correlation_regime=correlation_regime,
                
                # 驱动因素
                primary_drivers=primary_drivers,
                risk_factors=risk_factors,
                opportunities=opportunities,
                
                # 预测和建议
                short_term_outlook=short_term_outlook,
                medium_term_outlook=medium_term_outlook,
                recommended_positioning=positioning,
                hedge_suggestions=hedges,
                
                analysis_timestamp=datetime.now(),
                next_update=datetime.now() + timedelta(hours=4)
            )
            
        except Exception as e:
            self.logger.error(f"全球情绪分析失败: {e}")
            # 返回默认分析结果
            return self._create_default_analysis()
            
    def _analyze_volatility_sentiment(self, volatility_data: List[VolatilityIndicator]) -> float:
        """分析波动率情绪"""
        if not volatility_data:
            return 0.0
            
        vix_score = 0.0
        
        for indicator in volatility_data:
            if indicator.symbol == '^VIX':
                # VIX解读：低于20为低波动率，20-30为正常，30以上为高波动率
                if indicator.value < 20:
                    vix_score = 0.3  # 乐观
                elif indicator.value < 30:
                    vix_score = 0.0  # 中性
                else:
                    vix_score = -0.5  # 悲观
                    
                # 考虑百分位排名
                if indicator.percentile_rank > 80:
                    vix_score -= 0.2  # 更悲观
                elif indicator.percentile_rank < 20:
                    vix_score += 0.2  # 更乐观
                    
                break
                
        return max(-1.0, min(1.0, vix_score))
        
    async def _analyze_correlation_sentiment(self, data: Dict[str, Any]) -> float:
        """分析跨资产相关性情绪"""
        try:
            # 获取主要资产的历史数据计算相关性
            symbols = ['^GSPC', 'GLD', '^TNX', 'DX-Y.NYB']
            correlations = []
            
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    correlation = await self._calculate_correlation(symbol1, symbol2)
                    if correlation is not None:
                        correlations.append(abs(correlation))  # 使用绝对值
                        
            if not correlations:
                return 0.0
                
            avg_correlation = np.mean(correlations)
            
            # 高相关性通常表示恐慌情绪（所有资产同向变动）
            if avg_correlation > 0.7:
                return -0.4  # 恐慌
            elif avg_correlation > 0.5:
                return -0.2  # 轻微恐慌
            elif avg_correlation < 0.3:
                return 0.2   # 正常分化
            else:
                return 0.0   # 中性
                
        except Exception as e:
            self.logger.error(f"相关性分析失败: {e}")
            return 0.0
            
    def _analyze_news_sentiment(self, news_data: Dict[str, Any]) -> float:
        """分析新闻情绪"""
        news_articles = news_data.get('news_sentiment', [])
        
        if not news_articles:
            return 0.0
            
        # 加权平均新闻情绪
        total_weight = 0
        weighted_sentiment = 0
        
        for article in news_articles:
            weight = article.importance * article.market_relevance
            weighted_sentiment += article.sentiment_score * weight
            total_weight += weight
            
        if total_weight == 0:
            return 0.0
            
        return weighted_sentiment / total_weight
        
    async def _analyze_technical_sentiment(self, data: Dict[str, Any]) -> float:
        """分析技术面情绪"""
        try:
            # 分析主要指数的技术指标
            indices = data.get('global_indices', [])
            if not indices:
                return 0.0
                
            technical_scores = []
            
            for index in indices:
                # 获取历史数据计算技术指标
                hist_data = await self.yahoo_source.fetch_data(
                    [index.symbol], 
                    data_type='history',
                    period='3mo'
                )
                
                if hist_data.get('data'):
                    prices = [point.close for point in hist_data['data']]
                    
                    # 计算RSI
                    rsi = calculate_rsi(prices)
                    if rsi:
                        if rsi[-1] > 70:
                            technical_scores.append(-0.3)  # 超买
                        elif rsi[-1] < 30:
                            technical_scores.append(0.3)   # 超卖
                        else:
                            technical_scores.append(0.0)   # 中性
                            
            return np.mean(technical_scores) if technical_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"技术分析失败: {e}")
            return 0.0
            
    def _analyze_flight_to_quality(self, data: Dict[str, Any]) -> float:
        """分析避险流动情绪"""
        safe_haven_data = data.get('safe_haven_data', [])
        risk_assets_data = data.get('risk_assets_data', [])
        
        if not safe_haven_data or not risk_assets_data:
            return 0.0
            
        # 计算避险资产和风险资产的平均表现
        safe_haven_performance = np.mean([asset.change_percent or 0 for asset in safe_haven_data])
        risk_assets_performance = np.mean([asset.change_percent or 0 for asset in risk_assets_data])
        
        # 避险资产表现好于风险资产时，表示恐慌情绪
        performance_diff = safe_haven_performance - risk_assets_performance
        
        # 归一化到-1到1范围
        return max(-1.0, min(1.0, performance_diff / 5.0))
        
    def _calculate_weighted_sentiment(self, components: Dict[str, float]) -> float:
        """计算加权情绪分数"""
        weighted_sum = 0.0
        
        for component, score in components.items():
            weight = self.sentiment_weights.get(component, 0.0)
            weighted_sum += score * weight
            
        return max(-1.0, min(1.0, weighted_sum))
        
    def _determine_sentiment_type(self, sentiment_score: float) -> SentimentType:
        """确定情绪类型"""
        if sentiment_score > 0.3:
            return SentimentType.GREED
        elif sentiment_score < -0.3:
            return SentimentType.FEAR
        elif abs(sentiment_score) < 0.1:
            return SentimentType.NEUTRAL
        else:
            return SentimentType.UNCERTAINTY
            
    def _assess_risk_level(self, components: Dict[str, float]) -> RiskLevel:
        """评估风险级别"""
        # 基于各组件的恐慌程度评估风险
        fear_components = sum(1 for score in components.values() if score < -0.2)
        extreme_fear = sum(1 for score in components.values() if score < -0.5)
        
        if extreme_fear >= 2:
            return RiskLevel.EXTREME
        elif fear_components >= 3:
            return RiskLevel.HIGH
        elif fear_components >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
            
    def _calculate_asset_class_sentiments(self, data: Dict[str, Any]) -> Dict[str, float]:
        """计算各资产类别情绪"""
        indices = data.get('global_indices', [])
        
        # 计算股票情绪（基于指数表现）
        equity_performance = []
        for index in indices:
            if index.region in ['US', 'UK', 'DE', 'JP']:
                equity_performance.append(index.change_percent)
                
        equity_sentiment = np.mean(equity_performance) / 2.0 if equity_performance else 0.0
        
        # 债券情绪（基于收益率变化，简化处理）
        bond_sentiment = 0.0  # 需要债券数据
        
        # 商品情绪（基于避险资产表现）
        safe_haven_data = data.get('safe_haven_data', [])
        commodity_sentiment = 0.0
        for asset in safe_haven_data:
            if asset.symbol == 'GLD':  # 黄金
                commodity_sentiment = (asset.change_percent or 0) / 2.0
                break
                
        # 货币情绪（基于美元指数）
        currency_sentiment = 0.0
        for asset in safe_haven_data:
            if asset.symbol == 'DX-Y.NYB':  # 美元指数
                currency_sentiment = (asset.change_percent or 0) / 2.0
                break
                
        return {
            'equity': max(-1.0, min(1.0, equity_sentiment)),
            'bond': bond_sentiment,
            'commodity': max(-1.0, min(1.0, commodity_sentiment)),
            'currency': max(-1.0, min(1.0, currency_sentiment))
        }
        
    def _calculate_fear_greed_index(self, components: Dict[str, float]) -> float:
        """计算恐惧贪婪指数（0-100，50为中性）"""
        # 将-1到1的情绪分数转换为0-100的指数
        avg_sentiment = np.mean(list(components.values()))
        fear_greed = 50 + (avg_sentiment * 50)
        return max(0.0, min(100.0, fear_greed))
        
    def _determine_volatility_regime(self, volatility_data: List[VolatilityIndicator]) -> str:
        """确定波动率制度"""
        if not volatility_data:
            return "unknown"
            
        for indicator in volatility_data:
            if indicator.symbol == '^VIX':
                if indicator.value < 15:
                    return "low"
                elif indicator.value < 20:
                    return "normal"
                elif indicator.value < 30:
                    return "elevated"
                else:
                    return "extreme"
                    
        return "normal"
        
    def _determine_correlation_regime(self, correlation_sentiment: float) -> str:
        """确定相关性制度"""
        if correlation_sentiment < -0.3:
            return "crisis"
        elif correlation_sentiment < -0.1:
            return "elevated"
        else:
            return "normal"
            
    def _identify_primary_drivers(self, components: Dict[str, float], data: Dict[str, Any]) -> List[str]:
        """识别主要驱动因素"""
        drivers = []
        
        # 根据各组件的强度识别驱动因素
        if abs(components.get('volatility', 0)) > 0.3:
            drivers.append("波动率异常" if components['volatility'] < 0 else "波动率正常化")
            
        if abs(components.get('news_sentiment', 0)) > 0.2:
            drivers.append("新闻事件影响" if components['news_sentiment'] < 0 else "正面新闻推动")
            
        if abs(components.get('correlations', 0)) > 0.2:
            drivers.append("资产高度相关" if components['correlations'] < 0 else "资产正常分化")
            
        if abs(components.get('flight_to_quality', 0)) > 0.2:
            drivers.append("避险情绪上升" if components['flight_to_quality'] < 0 else "风险偏好回升")
            
        if not drivers:
            drivers.append("市场情绪相对稳定")
            
        return drivers[:3]  # 最多返回3个主要驱动因素
        
    def _identify_risk_factors(self, components: Dict[str, float], data: Dict[str, Any]) -> List[str]:
        """识别风险因素"""
        risks = []
        
        # 波动率风险
        if components.get('volatility', 0) < -0.3:
            risks.append("波动率飙升风险")
            
        # 相关性风险
        if components.get('correlations', 0) < -0.2:
            risks.append("资产相关性过高")
            
        # 新闻风险
        news_data = data.get('news_data', {})
        negative_news = sum(1 for article in news_data.get('news_sentiment', []) 
                          if article.sentiment_score < -0.3 and article.importance > 0.5)
        if negative_news >= 3:
            risks.append("重要负面新闻增多")
            
        # 地缘政治风险
        geo_events = data.get('news_data', {}).get('geopolitical_events', [])
        high_risk_events = sum(1 for event in geo_events if event.severity in [RiskLevel.HIGH, RiskLevel.EXTREME])
        if high_risk_events > 0:
            risks.append("地缘政治风险事件")
            
        return risks
        
    def _identify_opportunities(self, components: Dict[str, float], data: Dict[str, Any]) -> List[str]:
        """识别投资机会"""
        opportunities = []
        
        # 低波动率机会
        if components.get('volatility', 0) > 0.2:
            opportunities.append("低波动率环境，适合风险资产配置")
            
        # 正面情绪机会
        if components.get('news_sentiment', 0) > 0.2:
            opportunities.append("正面新闻推动，关注成长性资产")
            
        # 分化机会
        if components.get('correlations', 0) > 0.1:
            opportunities.append("资产分化良好，适合精选个股")
            
        # 避险机会
        if components.get('flight_to_quality', 0) < -0.2:
            opportunities.append("避险需求强烈，关注防御性配置")
            
        return opportunities
        
    def _generate_short_term_outlook(self, components: Dict[str, float]) -> str:
        """生成短期展望"""
        avg_sentiment = np.mean(list(components.values()))
        
        if avg_sentiment > 0.2:
            return "短期内市场情绪相对乐观，风险资产可能继续受益"
        elif avg_sentiment < -0.2:
            return "短期内恐慌情绪可能持续，建议关注避险资产"
        else:
            return "短期内市场情绪中性，预计震荡整理格局"
            
    def _generate_medium_term_outlook(self, components: Dict[str, float]) -> str:
        """生成中期展望"""
        volatility_level = abs(components.get('volatility', 0))
        
        if volatility_level > 0.3:
            return "中期内波动率可能保持高位，需要动态调整配置"
        else:
            return "中期内市场可能回归基本面驱动，关注经济数据表现"
            
    def _recommend_positioning(self, components: Dict[str, float], risk_level: RiskLevel) -> List[str]:
        """推荐仓位配置"""
        recommendations = []
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.EXTREME]:
            recommendations.extend([
                "减少风险资产敞口",
                "增加现金和短期债券配置",
                "考虑防御性股票"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "维持相对均衡配置",
                "适度降低杠杆水平",
                "关注高质量资产"
            ])
        else:
            recommendations.extend([
                "可适度增加风险资产配置",
                "关注成长性投资机会",
                "考虑新兴市场资产"
            ])
            
        return recommendations[:3]
        
    def _suggest_hedges(self, risk_level: RiskLevel, components: Dict[str, float]) -> List[str]:
        """建议对冲策略"""
        hedges = []
        
        if risk_level >= RiskLevel.MEDIUM:
            hedges.append("VIX看涨期权对冲波动率风险")
            
        if components.get('correlations', 0) < -0.2:
            hedges.append("跨资产对冲策略")
            
        if components.get('flight_to_quality', 0) < -0.2:
            hedges.append("增加黄金等避险资产配置")
            
        if not hedges:
            hedges.append("当前风险可控，无需特殊对冲")
            
        return hedges
        
    def _calculate_confidence(self, components: Dict[str, float], data: Dict[str, Any]) -> float:
        """计算分析置信度"""
        confidence = 0.0
        
        # 数据完整性
        data_completeness = 0.0
        total_sources = 5  # 预期数据源数量
        
        if data.get('global_indices'):
            data_completeness += 0.2
        if data.get('volatility_data'):
            data_completeness += 0.2
        if data.get('safe_haven_data'):
            data_completeness += 0.2
        if data.get('risk_assets_data'):
            data_completeness += 0.2
        if data.get('news_data', {}).get('news_sentiment'):
            data_completeness += 0.2
            
        confidence += data_completeness * 0.4
        
        # 信号一致性
        signal_consistency = 1.0 - np.std(list(components.values()))
        confidence += signal_consistency * 0.3
        
        # 数据新鲜度
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 16:  # 交易时间内
            confidence += 0.3
        else:
            confidence += 0.15
            
        return max(0.0, min(1.0, confidence))
        
    async def generate_report(self, analysis: SentimentAnalysis) -> str:
        """生成全球情绪分析报告"""
        
        timestamp = analysis.analysis_timestamp.strftime("%Y-%m-%d %H:%M")
        
        # 情绪描述
        sentiment_desc = {
            SentimentType.FEAR: "恐慌",
            SentimentType.GREED: "贪婪", 
            SentimentType.NEUTRAL: "中性",
            SentimentType.UNCERTAINTY: "不确定"
        }[analysis.overall_sentiment]
        
        # 风险级别描述
        risk_desc = {
            RiskLevel.LOW: "低风险",
            RiskLevel.MEDIUM: "中等风险",
            RiskLevel.HIGH: "高风险", 
            RiskLevel.EXTREME: "极端风险"
        }[analysis.risk_level]
        
        report = f"""
# 全球市场情绪分析报告

## 📊 综合情绪评估
**整体情绪**: {sentiment_desc}
**情绪分数**: {analysis.sentiment_score:+.3f} (-1恐慌, +1贪婪)
**分析置信度**: {analysis.confidence:.1%}
**风险级别**: {risk_desc}

## 🌍 分类资产情绪
• 股票市场: {analysis.equity_sentiment:+.2f}
• 债券市场: {analysis.bond_sentiment:+.2f}
• 商品市场: {analysis.commodity_sentiment:+.2f}
• 货币市场: {analysis.currency_sentiment:+.2f}

## 📈 市场状态指标
• 恐惧贪婪指数: {analysis.fear_greed_index:.0f}/100
• 波动率环境: {analysis.volatility_regime}
• 相关性状态: {analysis.correlation_regime}

## 🔍 主要驱动因素
{chr(10).join(f"• {driver}" for driver in analysis.primary_drivers)}

## ⚠️ 风险因素
{chr(10).join(f"• {risk}" for risk in analysis.risk_factors) if analysis.risk_factors else "• 当前未识别出重大风险因素"}

## 💡 投资机会
{chr(10).join(f"• {opp}" for opp in analysis.opportunities) if analysis.opportunities else "• 市场机会有限，建议谨慎操作"}

## 🔮 市场展望
**短期预期**: {analysis.short_term_outlook}
**中期预期**: {analysis.medium_term_outlook}

## 💼 配置建议
{chr(10).join(f"• {rec}" for rec in analysis.recommended_positioning)}

## 🛡️ 对冲建议
{chr(10).join(f"• {hedge}" for hedge in analysis.hedge_suggestions)}

---
*报告生成时间: {timestamp}*
*下次更新: {analysis.next_update.strftime("%Y-%m-%d %H:%M")}*
"""
        
        return report
        
    # 辅助方法
    def _calculate_percentile_rank(self, current_value: float, historical_data: List[Any]) -> float:
        """计算当前值在历史数据中的百分位排名"""
        if not historical_data:
            return 50.0  # 默认中位数
            
        values = [point.close for point in historical_data if hasattr(point, 'close')]
        if not values:
            return 50.0
            
        values.append(current_value)
        values.sort()
        
        rank = values.index(current_value) / len(values) * 100
        return rank
        
    def _interpret_volatility(self, value: float, percentile_rank: float) -> str:
        """解读波动率指标"""
        if value < 15:
            return "极低波动率，市场过度自满"
        elif value < 20:
            return "低波动率，市场情绪稳定"
        elif value < 30:
            return "正常波动率，市场运行健康"
        elif value < 40:
            return "高波动率，市场存在担忧"
        else:
            return "极高波动率，市场恐慌情绪严重"
            
    def _get_volatility_name(self, symbol: str) -> str:
        """获取波动率指标名称"""
        names = {
            '^VIX': '标普500波动率指数',
            '^VVIX': 'VIX波动率指数',
            '^VSTOXX': '欧洲股指波动率'
        }
        return names.get(symbol, symbol)
        
    async def _calculate_correlation(self, symbol1: str, symbol2: str, period: str = '3mo') -> Optional[float]:
        """计算两个资产的相关性"""
        try:
            # 获取历史数据
            data1 = await self.yahoo_source.fetch_data([symbol1], data_type='history', period=period)
            data2 = await self.yahoo_source.fetch_data([symbol2], data_type='history', period=period)
            
            prices1 = [point.close for point in data1.get('data', [])]
            prices2 = [point.close for point in data2.get('data', [])]
            
            if len(prices1) < 10 or len(prices2) < 10:
                return None
                
            # 计算收益率
            returns1 = np.diff(prices1) / prices1[:-1]
            returns2 = np.diff(prices2) / prices2[:-1]
            
            # 确保长度一致
            min_length = min(len(returns1), len(returns2))
            returns1 = returns1[:min_length]
            returns2 = returns2[:min_length]
            
            # 计算相关系数
            correlation, _ = pearsonr(returns1, returns2)
            
            return correlation if not np.isnan(correlation) else None
            
        except Exception as e:
            self.logger.error(f"计算相关性失败 {symbol1}-{symbol2}: {e}")
            return None
            
    def _create_default_analysis(self) -> SentimentAnalysis:
        """创建默认分析结果"""
        return SentimentAnalysis(
            overall_sentiment=SentimentType.NEUTRAL,
            sentiment_score=0.0,
            confidence=0.5,
            risk_level=RiskLevel.MEDIUM,
            equity_sentiment=0.0,
            bond_sentiment=0.0,
            commodity_sentiment=0.0,
            currency_sentiment=0.0,
            fear_greed_index=50.0,
            volatility_regime="normal",
            correlation_regime="normal",
            primary_drivers=["数据不足，无法分析"],
            risk_factors=["数据获取异常"],
            opportunities=["等待数据完整后再评估"],
            short_term_outlook="数据不足，暂无预测",
            medium_term_outlook="数据不足，暂无预测",
            recommended_positioning=["等待数据完整"],
            hedge_suggestions=["数据不足时建议保守操作"],
            analysis_timestamp=datetime.now(),
            next_update=datetime.now() + timedelta(hours=1)
        )
        
    async def cleanup(self) -> None:
        """清理资源"""
        if self.yahoo_source:
            await self.yahoo_source.cleanup()
        if self.news_source:
            await self.news_source.cleanup()
            
        self.logger.info("GlobalSentiment Agent清理完成")
        
    def get_dependencies(self) -> List[str]:
        """获取依赖的其他Agent"""
        return []  # 独立Agent，不依赖其他Agent 