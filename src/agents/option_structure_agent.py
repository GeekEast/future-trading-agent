"""
期权市场结构分析Agent
分析期权链、最大痛点、Gamma敞口、支撑阻力、波动率曲面
"""

import asyncio
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from scipy.stats import norm
import math

from .base_agent import BaseAgent
from data_sources.yahoo_finance import YahooFinanceDataSource
from models.option_data import (
    Option, OptionChain, OptionType, MaxPainAnalysis, 
    GammaExposure, SupportResistance, VolatilitySurface,
    OptionMetrics, OptionAnalysisResult
)
from utils.helpers import calculate_volatility


class OptionStructureAgent(BaseAgent):
    """期权市场结构分析Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("OptionStructure", config)
        
        # 数据源配置
        self.yahoo_source = None
        
        # 支持的期权标的
        self.option_symbols = [
            'SPY',   # S&P 500 ETF
            'QQQ',   # NASDAQ 100 ETF  
            'IWM',   # Russell 2000 ETF
            'AAPL',  # Apple
            'MSFT',  # Microsoft
            'TSLA',  # Tesla
            'NVDA',  # NVIDIA
            'META',  # Meta
            'GOOGL', # Google
            'AMZN'   # Amazon
        ]
        
        # 无风险利率（用于Black-Scholes计算）
        self.risk_free_rate = 0.05  # 5%，可从实际利率获取
        
        # 分析参数
        self.analysis_config = {
            'min_volume': 10,           # 最小成交量过滤
            'min_open_interest': 50,    # 最小持仓量过滤
            'gamma_threshold': 0.01,    # Gamma阈值
            'max_days_to_expiry': 60,   # 最大到期天数
            'strike_range_pct': 0.15    # 行权价范围（相对价格）
        }
        
    async def initialize(self) -> None:
        """初始化OptionStructure Agent"""
        
        # 初始化数据源
        self.yahoo_source = YahooFinanceDataSource(self.config)
        await self.yahoo_source.initialize()
        
        self.logger.info("OptionStructure Agent初始化完成")
        
    async def fetch_data(self, symbol: str = 'SPY') -> Dict[str, Any]:
        """获取期权市场结构分析所需数据"""
        
        try:
            if symbol not in self.option_symbols:
                self.logger.warning(f"标的 {symbol} 可能没有活跃期权")
                
            # 获取标的价格和期权链
            tasks = [
                self._fetch_underlying_data(symbol),
                self._fetch_option_chain_data(symbol)
            ]
            
            underlying_data, option_chains = await asyncio.gather(*tasks, return_exceptions=True)
            
            if isinstance(underlying_data, Exception):
                self.logger.error(f"获取标的数据失败: {underlying_data}")
                underlying_data = {}
                
            if isinstance(option_chains, Exception):
                self.logger.error(f"获取期权链数据失败: {option_chains}")
                option_chains = []
                
            # 计算期权数量统计
            total_options = sum(len(chain.options) for chain in option_chains)
            call_count = sum(len(chain.get_calls()) for chain in option_chains)
            put_count = sum(len(chain.get_puts()) for chain in option_chains)
            
            return {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'underlying_data': underlying_data,
                'option_chains': option_chains,
                'total_options': total_options,
                'call_count': call_count,
                'put_count': put_count,
                'expiry_count': len(option_chains)
            }
            
        except Exception as e:
            self.logger.error(f"获取期权数据失败: {e}")
            return {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'error': str(e)
            }
            
    async def _fetch_underlying_data(self, symbol: str) -> Dict[str, Any]:
        """获取标的资产数据"""
        try:
            # 获取当前价格
            quote_data = await self.yahoo_source.fetch_data([symbol], data_type='quote')
            
            # 获取历史数据用于计算已实现波动率
            hist_data = await self.yahoo_source.fetch_data(
                [symbol], 
                data_type='history', 
                period='1mo'
            )
            
            current_price = 0.0
            realized_vol = 0.0
            
            if quote_data.get('data'):
                current_price = quote_data['data'][0].price
                
            if hist_data.get('data'):
                prices = [point.close for point in hist_data['data']]
                realized_vol = calculate_volatility(prices, periods=252)
                
            return {
                'symbol': symbol,
                'current_price': current_price,
                'realized_volatility': realized_vol,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"获取标的数据失败 {symbol}: {e}")
            return {}
            
    async def _fetch_option_chain_data(self, symbol: str) -> List[OptionChain]:
        """获取期权链数据"""
        try:
            option_data = await self.yahoo_source.fetch_data([symbol], data_type='options')
            
            if not option_data.get('data'):
                return []
                
            # 过滤和清理期权链
            filtered_chains = []
            for chain in option_data['data']:
                if isinstance(chain, OptionChain):
                    # 过滤期权（成交量和持仓量）
                    filtered_options = self._filter_options(chain.options)
                    if filtered_options:
                        chain.options = filtered_options
                        filtered_chains.append(chain)
                        
            return filtered_chains
            
        except Exception as e:
            self.logger.error(f"获取期权链数据失败 {symbol}: {e}")
            return []
            
    def _filter_options(self, options: List[Option]) -> List[Option]:
        """过滤期权数据"""
        filtered = []
        
        for option in options:
            # 检查成交量和持仓量
            volume = option.volume or 0
            oi = option.open_interest or 0
            
            if (volume >= self.analysis_config['min_volume'] or 
                oi >= self.analysis_config['min_open_interest']):
                
                # 检查到期时间
                days_to_expiry = option.days_to_expiration()
                if 0 < days_to_expiry <= self.analysis_config['max_days_to_expiry']:
                    filtered.append(option)
                    
        return filtered
        
    async def analyze(self, data: Dict[str, Any]) -> OptionAnalysisResult:
        """分析期权市场结构"""
        
        try:
            symbol = data.get('symbol', 'SPY')
            underlying_data = data.get('underlying_data', {})
            option_chains = data.get('option_chains', [])
            
            if not option_chains:
                return self._create_default_analysis(symbol)
                
            current_price = underlying_data.get('current_price', 0)
            if current_price <= 0:
                return self._create_default_analysis(symbol)
                
            # 执行各项分析
            max_pain = self._analyze_max_pain(option_chains, current_price)
            gamma_exposure = self._analyze_gamma_exposure(option_chains, current_price)
            support_resistance = self._analyze_support_resistance(option_chains, current_price)
            volatility_surface = self._analyze_volatility_surface(option_chains)
            metrics = self._calculate_option_metrics(option_chains, underlying_data)
            
            # 确定交易区间和市场展望
            trading_range = self._calculate_trading_range(
                max_pain, gamma_exposure, support_resistance, current_price
            )
            
            risk_level = self._assess_risk_level(gamma_exposure, metrics)
            market_outlook = self._generate_market_outlook(
                max_pain, gamma_exposure, support_resistance, metrics, current_price
            )
            
            return OptionAnalysisResult(
                underlying=symbol,
                underlying_price=current_price,
                analysis_time=datetime.now(),
                option_chains=option_chains,
                max_pain=max_pain,
                gamma_exposure=gamma_exposure,
                support_resistance=support_resistance,
                volatility_surface=volatility_surface,
                metrics=metrics,
                trading_range=trading_range,
                risk_level=risk_level,
                market_outlook=market_outlook
            )
            
        except Exception as e:
            self.logger.error(f"期权结构分析失败: {e}")
            return self._create_default_analysis(data.get('symbol', 'SPY'))
            
    def _analyze_max_pain(self, option_chains: List[OptionChain], current_price: float) -> MaxPainAnalysis:
        """分析最大痛点"""
        
        # 收集所有期权
        all_options = []
        for chain in option_chains:
            all_options.extend(chain.options)
            
        if not all_options:
            return MaxPainAnalysis(
                underlying=option_chains[0].underlying if option_chains else "UNKNOWN",
                expiration_date=date.today(),
                max_pain_price=current_price,
                total_pain_at_max=0,
                pain_by_strike={},
                call_put_ratio=1.0
            )
            
        # 获取所有行权价
        strikes = sorted(set(opt.strike for opt in all_options))
        
        # 计算各行权价的痛苦值
        pain_by_strike = {}
        
        for strike in strikes:
            call_pain = 0
            put_pain = 0
            
            for option in all_options:
                oi = option.open_interest or 0
                
                if option.option_type == OptionType.CALL and strike > option.strike:
                    call_pain += (strike - option.strike) * oi
                elif option.option_type == OptionType.PUT and strike < option.strike:
                    put_pain += (option.strike - strike) * oi
                    
            pain_by_strike[strike] = call_pain + put_pain
            
        # 找到最大痛点
        if pain_by_strike:
            max_pain_price = min(pain_by_strike.keys(), key=lambda k: pain_by_strike[k])
            total_pain_at_max = pain_by_strike[max_pain_price]
        else:
            max_pain_price = current_price
            total_pain_at_max = 0
            
        # 计算Call/Put比率
        call_oi = sum(opt.open_interest or 0 for opt in all_options if opt.option_type == OptionType.CALL)
        put_oi = sum(opt.open_interest or 0 for opt in all_options if opt.option_type == OptionType.PUT)
        call_put_ratio = call_oi / max(put_oi, 1)
        
        return MaxPainAnalysis(
            underlying=option_chains[0].underlying,
            expiration_date=option_chains[0].expiration_date,
            max_pain_price=max_pain_price,
            total_pain_at_max=total_pain_at_max,
            pain_by_strike=pain_by_strike,
            call_put_ratio=call_put_ratio
        )
        
    def _analyze_gamma_exposure(self, option_chains: List[OptionChain], current_price: float) -> GammaExposure:
        """分析Gamma敞口"""
        
        all_options = []
        for chain in option_chains:
            all_options.extend(chain.options)
            
        if not all_options:
            return GammaExposure(
                underlying=option_chains[0].underlying if option_chains else "UNKNOWN",
                underlying_price=current_price,
                net_gamma=0,
                gamma_by_strike={},
                zero_gamma_level=current_price
            )
            
        # 计算每个期权的Gamma（如果没有则估算）
        gamma_by_strike = defaultdict(float)
        net_gamma = 0
        
        for option in all_options:
            gamma = option.gamma
            
            # 如果没有Gamma数据，使用Black-Scholes估算
            if gamma is None:
                gamma = self._estimate_gamma(option, current_price)
                
            if gamma is not None:
                oi = option.open_interest or 0
                
                # 假设做市商持有空头（实际情况可能更复杂）
                dealer_gamma = -gamma * oi * 100  # 每张合约100股
                
                gamma_by_strike[option.strike] += dealer_gamma
                net_gamma += dealer_gamma
                
        # 寻找零Gamma水平
        zero_gamma_level = self._find_zero_gamma_level(gamma_by_strike, current_price)
        
        # 确定正负Gamma区间
        strikes = sorted(gamma_by_strike.keys())
        positive_range = []
        negative_range = []
        
        for strike in strikes:
            if gamma_by_strike[strike] > 0:
                positive_range.append(strike)
            elif gamma_by_strike[strike] < 0:
                negative_range.append(strike)
                
        return GammaExposure(
            underlying=option_chains[0].underlying,
            underlying_price=current_price,
            net_gamma=net_gamma,
            gamma_by_strike=dict(gamma_by_strike),
            zero_gamma_level=zero_gamma_level,
            positive_gamma_range=positive_range if positive_range else None,
            negative_gamma_range=negative_range if negative_range else None
        )
        
    def _estimate_gamma(self, option: Option, underlying_price: float) -> Optional[float]:
        """使用Black-Scholes模型估算Gamma"""
        try:
            S = underlying_price
            K = option.strike
            T = option.time_to_expiration()
            r = self.risk_free_rate
            sigma = option.implied_volatility or 0.2  # 默认20%波动率
            
            if T <= 0 or sigma <= 0:
                return None
                
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            
            # Gamma公式
            gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
            
            return gamma
            
        except Exception as e:
            self.logger.warning(f"Gamma估算失败: {e}")
            return None
            
    def _find_zero_gamma_level(self, gamma_by_strike: Dict[float, float], current_price: float) -> Optional[float]:
        """寻找零Gamma水平"""
        if not gamma_by_strike:
            return current_price
            
        strikes = sorted(gamma_by_strike.keys())
        
        # 寻找Gamma从正变负或从负变正的点
        for i in range(len(strikes) - 1):
            gamma1 = gamma_by_strike[strikes[i]]
            gamma2 = gamma_by_strike[strikes[i + 1]]
            
            if gamma1 * gamma2 < 0:  # 符号相反
                # 线性插值
                ratio = abs(gamma1) / (abs(gamma1) + abs(gamma2))
                zero_level = strikes[i] + ratio * (strikes[i + 1] - strikes[i])
                return zero_level
                
        return None
        
    def _analyze_support_resistance(self, option_chains: List[OptionChain], current_price: float) -> SupportResistance:
        """分析期权支撑阻力"""
        
        all_options = []
        for chain in option_chains:
            all_options.extend(chain.options)
            
        if not all_options:
            return SupportResistance(
                underlying=option_chains[0].underlying if option_chains else "UNKNOWN",
                support_levels=[],
                resistance_levels=[],
                strength_scores={}
            )
            
        # 按行权价分组计算强度
        strike_strength = defaultdict(float)
        
        for option in all_options:
            oi = option.open_interest or 0
            volume = option.volume or 0
            
            # 使用持仓量和成交量计算强度
            strength = oi * 0.7 + volume * 0.3
            strike_strength[option.strike] += strength
            
        # 识别支撑和阻力位
        strikes = sorted(strike_strength.keys())
        support_levels = []
        resistance_levels = []
        
        # 当前价格下方为支撑，上方为阻力
        for strike in strikes:
            if strike < current_price * 0.98:  # 2%容忍度
                support_levels.append(strike)
            elif strike > current_price * 1.02:
                resistance_levels.append(strike)
                
        # 按强度排序，取前5个
        support_levels = sorted(support_levels, key=lambda x: strike_strength[x], reverse=True)[:5]
        resistance_levels = sorted(resistance_levels, key=lambda x: strike_strength[x], reverse=True)[:5]
        
        # 识别Call墙和Put墙
        call_wall = None
        put_wall = None
        
        # Call墙：大量Call持仓的阻力位
        call_strikes = defaultdict(float)
        put_strikes = defaultdict(float)
        
        for option in all_options:
            oi = option.open_interest or 0
            if option.option_type == OptionType.CALL:
                call_strikes[option.strike] += oi
            else:
                put_strikes[option.strike] += oi
                
        if call_strikes:
            call_wall = max(call_strikes.keys(), key=lambda k: call_strikes[k])
            
        if put_strikes:
            put_wall = max(put_strikes.keys(), key=lambda k: put_strikes[k])
            
        return SupportResistance(
            underlying=option_chains[0].underlying,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            call_wall=call_wall,
            put_wall=put_wall,
            strength_scores=dict(strike_strength)
        )
        
    def _analyze_volatility_surface(self, option_chains: List[OptionChain]) -> VolatilitySurface:
        """分析波动率曲面"""
        
        if not option_chains:
            return VolatilitySurface(
                underlying="UNKNOWN",
                surface_data={},
                term_structure={},
                skew_by_expiry={}
            )
            
        surface_data = {}
        term_structure = {}
        skew_by_expiry = {}
        
        for chain in option_chains:
            expiry_str = chain.expiration_date.strftime('%Y-%m-%d')
            
            # 收集该到期日的隐含波动率数据
            strike_iv_map = {}
            ivs = []
            
            for option in chain.options:
                if option.implied_volatility and option.implied_volatility > 0:
                    strike_iv_map[option.strike] = option.implied_volatility
                    ivs.append(option.implied_volatility)
                    
            if strike_iv_map:
                surface_data[expiry_str] = strike_iv_map
                
                # 计算平值波动率（ATM IV）
                underlying_price = chain.underlying_price
                strikes = list(strike_iv_map.keys())
                atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
                term_structure[expiry_str] = strike_iv_map[atm_strike]
                
                # 计算偏斜（最高IV - 最低IV）
                if len(ivs) > 1:
                    skew_by_expiry[expiry_str] = max(ivs) - min(ivs)
                else:
                    skew_by_expiry[expiry_str] = 0
                    
        return VolatilitySurface(
            underlying=option_chains[0].underlying,
            surface_data=surface_data,
            term_structure=term_structure,
            skew_by_expiry=skew_by_expiry
        )
        
    def _calculate_option_metrics(self, option_chains: List[OptionChain], underlying_data: Dict[str, Any]) -> OptionMetrics:
        """计算期权市场指标"""
        
        all_options = []
        for chain in option_chains:
            all_options.extend(chain.options)
            
        if not all_options:
            return OptionMetrics(
                underlying=option_chains[0].underlying if option_chains else "UNKNOWN",
                timestamp=datetime.now(),
                put_call_ratio=1.0,
                put_call_volume_ratio=1.0,
                put_call_oi_ratio=1.0
            )
            
        # 分离Call和Put
        calls = [opt for opt in all_options if opt.option_type == OptionType.CALL]
        puts = [opt for opt in all_options if opt.option_type == OptionType.PUT]
        
        # 计算基础比率
        put_count = len(puts)
        call_count = len(calls)
        put_call_ratio = put_count / max(call_count, 1)
        
        # 成交量比率
        put_volume = sum(opt.volume or 0 for opt in puts)
        call_volume = sum(opt.volume or 0 for opt in calls)
        put_call_volume_ratio = put_volume / max(call_volume, 1)
        
        # 持仓量比率
        put_oi = sum(opt.open_interest or 0 for opt in puts)
        call_oi = sum(opt.open_interest or 0 for opt in calls)
        put_call_oi_ratio = put_oi / max(call_oi, 1)
        
        # 计算已实现波动率
        realized_vol = underlying_data.get('realized_volatility', 0)
        
        # 计算隐含波动率统计
        ivs = [opt.implied_volatility for opt in all_options if opt.implied_volatility]
        iv_rank = None
        iv_percentile = None
        
        if ivs:
            current_iv = np.mean(ivs)
            # 简化的IV排名（需要历史数据来计算真实排名）
            iv_rank = min(100, max(0, current_iv * 100))
            iv_percentile = iv_rank
            
        return OptionMetrics(
            underlying=option_chains[0].underlying,
            timestamp=datetime.now(),
            put_call_ratio=put_call_ratio,
            put_call_volume_ratio=put_call_volume_ratio,
            put_call_oi_ratio=put_call_oi_ratio,
            realized_volatility=realized_vol,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile
        )
        
    def _calculate_trading_range(
        self, 
        max_pain: MaxPainAnalysis, 
        gamma_exposure: GammaExposure, 
        support_resistance: SupportResistance,
        current_price: float
    ) -> List[float]:
        """计算预期交易区间"""
        
        levels = []
        
        # 添加最大痛点
        levels.append(max_pain.max_pain_price)
        
        # 添加零Gamma水平
        if gamma_exposure.zero_gamma_level:
            levels.append(gamma_exposure.zero_gamma_level)
            
        # 添加主要支撑阻力位
        if support_resistance.support_levels:
            levels.extend(support_resistance.support_levels[:2])
            
        if support_resistance.resistance_levels:
            levels.extend(support_resistance.resistance_levels[:2])
            
        # 过滤和排序
        levels = [level for level in levels if 0.8 * current_price <= level <= 1.2 * current_price]
        levels = sorted(set(levels))
        
        if len(levels) >= 2:
            return [min(levels), max(levels)]
        else:
            # 默认±5%
            return [current_price * 0.95, current_price * 1.05]
            
    def _assess_risk_level(self, gamma_exposure: GammaExposure, metrics: OptionMetrics) -> str:
        """评估风险级别"""
        
        risk_score = 0
        
        # Gamma环境评估
        if gamma_exposure.is_positive_gamma_environment():
            risk_score += 1  # 正Gamma环境风险较低
        else:
            risk_score += 3  # 负Gamma环境风险较高
            
        # Put/Call比率评估
        if metrics.put_call_ratio > 1.5:
            risk_score += 2  # 过度看跌
        elif metrics.put_call_ratio < 0.5:
            risk_score += 2  # 过度看涨
        else:
            risk_score += 1  # 相对平衡
            
        # 隐含波动率评估
        if metrics.iv_rank and metrics.iv_rank > 80:
            risk_score += 2  # 高波动率环境
        elif metrics.iv_rank and metrics.iv_rank < 20:
            risk_score += 1  # 低波动率环境
            
        # 风险级别判定
        if risk_score <= 3:
            return "低风险"
        elif risk_score <= 5:
            return "中等风险"
        else:
            return "高风险"
            
    def _generate_market_outlook(
        self,
        max_pain: MaxPainAnalysis,
        gamma_exposure: GammaExposure,
        support_resistance: SupportResistance,
        metrics: OptionMetrics,
        current_price: float
    ) -> str:
        """生成市场展望"""
        
        outlook_factors = []
        
        # 最大痛点分析
        pain_distance = abs(current_price - max_pain.max_pain_price) / current_price
        if pain_distance > 0.05:
            direction = "上涨" if current_price < max_pain.max_pain_price else "下跌"
            outlook_factors.append(f"最大痛点在{max_pain.max_pain_price:.2f}，预期价格向{direction}")
            
        # Gamma环境分析
        if gamma_exposure.is_positive_gamma_environment():
            outlook_factors.append("正Gamma环境，价格波动可能受到抑制")
        else:
            outlook_factors.append("负Gamma环境，价格波动可能被放大")
            
        # 支撑阻力分析
        nearest_support = support_resistance.get_nearest_support(current_price)
        nearest_resistance = support_resistance.get_nearest_resistance(current_price)
        
        if nearest_support:
            outlook_factors.append(f"下方支撑位{nearest_support:.2f}")
        if nearest_resistance:
            outlook_factors.append(f"上方阻力位{nearest_resistance:.2f}")
            
        # Put/Call比率分析
        if metrics.put_call_ratio > 1.2:
            outlook_factors.append("看跌情绪较浓，可能存在反弹机会")
        elif metrics.put_call_ratio < 0.8:
            outlook_factors.append("看涨情绪较浓，需要注意回调风险")
            
        return "；".join(outlook_factors) if outlook_factors else "市场展望中性"
        
    async def generate_report(self, analysis: OptionAnalysisResult) -> str:
        """生成期权结构分析报告"""
        
        timestamp = analysis.analysis_time.strftime("%Y-%m-%d %H:%M")
        key_levels = analysis.get_key_levels()
        trading_summary = analysis.get_trading_summary()
        
        report = f"""
# 期权市场结构分析报告

## 📊 基本信息
**标的资产**: {analysis.underlying}
**当前价格**: ${analysis.underlying_price:.2f}
**分析时间**: {timestamp}
**风险级别**: {analysis.risk_level}

## 🎯 关键价位分析
"""
        
        if key_levels:
            for level_name, level_value in key_levels.items():
                distance = (level_value - analysis.underlying_price) / analysis.underlying_price * 100
                report += f"• {level_name}: ${level_value:.2f} ({distance:+.1f}%)\n"
        else:
            report += "• 暂无明确关键价位\n"
            
        report += f"""
## 📈 期权结构概览
• **最大痛点**: ${analysis.max_pain.max_pain_price:.2f}
• **Call/Put比率**: {analysis.max_pain.call_put_ratio:.2f}
• **净Gamma敞口**: {analysis.gamma_exposure.net_gamma:,.0f}
• **Gamma环境**: {'正Gamma' if analysis.gamma_exposure.is_positive_gamma_environment() else '负Gamma'}

## 📊 支撑阻力分析
**支撑位**: {', '.join(f'${level:.2f}' for level in analysis.support_resistance.support_levels[:3]) if analysis.support_resistance.support_levels else '暂无明确支撑'}
**阻力位**: {', '.join(f'${level:.2f}' for level in analysis.support_resistance.resistance_levels[:3]) if analysis.support_resistance.resistance_levels else '暂无明确阻力'}
"""

        if analysis.support_resistance.call_wall:
            report += f"**Call墙**: ${analysis.support_resistance.call_wall:.2f}\n"
        if analysis.support_resistance.put_wall:
            report += f"**Put墙**: ${analysis.support_resistance.put_wall:.2f}\n"
            
        report += f"""
## 📊 期权指标
• **Put/Call持仓比**: {analysis.metrics.put_call_oi_ratio:.2f}
• **Put/Call成交比**: {analysis.metrics.put_call_volume_ratio:.2f}
• **情绪评分**: {analysis.metrics.get_sentiment_score():.0f}/100
"""

        if analysis.metrics.iv_rank:
            report += f"• **隐含波动率排名**: {analysis.metrics.iv_rank:.0f}%\n"
            
        report += f"""
## 🎯 预期交易区间
"""
        if analysis.trading_range and len(analysis.trading_range) == 2:
            range_low, range_high = analysis.trading_range
            range_width = (range_high - range_low) / analysis.underlying_price * 100
            report += f"**预期区间**: ${range_low:.2f} - ${range_high:.2f} (宽度: {range_width:.1f}%)\n"
        else:
            report += "**预期区间**: 暂无明确预期\n"
            
        report += f"""
## 🔮 市场展望
{analysis.market_outlook}

## 💡 交易建议
"""
        
        # 生成交易建议
        suggestions = self._generate_trading_suggestions(analysis)
        for suggestion in suggestions:
            report += f"• {suggestion}\n"
            
        report += f"""
---
*报告生成时间: {timestamp}*
*数据来源: 期权链分析*
"""
        
        return report
        
    def _generate_trading_suggestions(self, analysis: OptionAnalysisResult) -> List[str]:
        """生成交易建议"""
        suggestions = []
        
        current_price = analysis.underlying_price
        max_pain = analysis.max_pain.max_pain_price
        
        # 基于最大痛点的建议
        pain_distance = abs(current_price - max_pain) / current_price
        if pain_distance > 0.03:
            if current_price < max_pain:
                suggestions.append(f"当前价格低于最大痛点，可能有向上修复至${max_pain:.2f}的动力")
            else:
                suggestions.append(f"当前价格高于最大痛点，可能面临向下调整至${max_pain:.2f}的压力")
                
        # 基于Gamma环境的建议
        if analysis.gamma_exposure.is_positive_gamma_environment():
            suggestions.append("正Gamma环境下，价格波动可能受限，适合卖出策略")
        else:
            suggestions.append("负Gamma环境下，价格波动可能放大，需要注意风险控制")
            
        # 基于支撑阻力的建议
        nearest_support = analysis.support_resistance.get_nearest_support(current_price)
        nearest_resistance = analysis.support_resistance.get_nearest_resistance(current_price)
        
        if nearest_support and (current_price - nearest_support) / current_price < 0.05:
            suggestions.append(f"接近支撑位${nearest_support:.2f}，可考虑逢低买入")
            
        if nearest_resistance and (nearest_resistance - current_price) / current_price < 0.05:
            suggestions.append(f"接近阻力位${nearest_resistance:.2f}，可考虑获利了结")
            
        # 基于Put/Call比率的建议
        if analysis.metrics.put_call_ratio > 1.2:
            suggestions.append("看跌情绪过度，可能存在反向交易机会")
        elif analysis.metrics.put_call_ratio < 0.8:
            suggestions.append("看涨情绪过度，建议保持谨慎")
            
        if not suggestions:
            suggestions.append("当前期权结构相对均衡，建议观望等待明确信号")
            
        return suggestions
        
    def _create_default_analysis(self, symbol: str) -> OptionAnalysisResult:
        """创建默认分析结果"""
        
        return OptionAnalysisResult(
            underlying=symbol,
            underlying_price=0.0,
            analysis_time=datetime.now(),
            option_chains=[],
            max_pain=MaxPainAnalysis(
                underlying=symbol,
                expiration_date=date.today(),
                max_pain_price=0.0,
                total_pain_at_max=0,
                pain_by_strike={},
                call_put_ratio=1.0
            ),
            gamma_exposure=GammaExposure(
                underlying=symbol,
                underlying_price=0.0,
                net_gamma=0,
                gamma_by_strike={}
            ),
            support_resistance=SupportResistance(
                underlying=symbol,
                support_levels=[],
                resistance_levels=[],
                strength_scores={}
            ),
            volatility_surface=VolatilitySurface(
                underlying=symbol,
                surface_data={},
                term_structure={},
                skew_by_expiry={}
            ),
            metrics=OptionMetrics(
                underlying=symbol,
                timestamp=datetime.now(),
                put_call_ratio=1.0,
                put_call_volume_ratio=1.0,
                put_call_oi_ratio=1.0
            ),
            trading_range=None,
            risk_level="数据不足",
            market_outlook="缺乏期权数据，无法提供市场展望"
        )
        
    async def cleanup(self) -> None:
        """清理资源"""
        if self.yahoo_source:
            await self.yahoo_source.cleanup()
            
        self.logger.info("OptionStructure Agent清理完成")
        
    def get_dependencies(self) -> List[str]:
        """获取依赖的其他Agent"""
        return []  # 独立Agent，不依赖其他Agent 