#!/usr/bin/env python3
"""
期货交易AI Agent主程序
基于LangChain+LangGraph的盘前分析系统
"""

import asyncio
import click
from loguru import logger
from pathlib import Path
from datetime import datetime

from utils.config import load_config
from utils.logger import setup_logger
from workflows.comprehensive_analysis_workflow import ComprehensiveAnalysisWorkflow
from agents.macro_policy_agent import MacroPolicyAgent
from agents.global_sentiment_agent import GlobalSentimentAgent
from agents.option_structure_agent import OptionStructureAgent


@click.command()
@click.option('--config', '-c', default='config/settings.yaml', help='配置文件路径')
@click.option('--debug', '-d', is_flag=True, help='调试模式')
@click.option('--agent', '-a', type=click.Choice(['macro', 'sentiment', 'option', 'all']), 
              default='all', help='运行指定的Agent')
@click.option('--symbol', '-s', default='SPY', help='分析目标标的')
@click.option('--output', '-o', help='输出报告文件路径')
def main(config, debug, agent, symbol, output):
    """期货交易AI Agent主程序"""
    
    # 加载配置
    settings = load_config(config)
    
    # 设置日志
    setup_logger(debug=debug)
    
    logger.info("🚀 启动期货交易AI Agent...")
    logger.info(f"配置文件: {config}")
    logger.info(f"调试模式: {debug}")
    logger.info(f"运行Agent: {agent}")
    logger.info(f"分析标的: {symbol}")
    
    # 运行分析
    asyncio.run(run_analysis(settings, agent, symbol, output, debug))


async def run_analysis(settings: dict, agent: str, symbol: str, output: str = None, debug: bool = False):
    """运行分析逻辑"""
    try:
        if agent == "all":
            # 运行综合分析工作流
            await run_comprehensive_analysis(settings, symbol, output)
        elif agent == "macro":
            # 运行单独的宏观Agent
            await run_single_agent_analysis(settings, MacroPolicyAgent, "宏观政策分析", output)
        elif agent == "sentiment":
            # 运行单独的情绪Agent
            await run_single_agent_analysis(settings, GlobalSentimentAgent, "全球情绪分析", output)
        elif agent == "option":
            # 运行单独的期权Agent
            await run_single_agent_analysis(settings, OptionStructureAgent, "期权结构分析", output, symbol)
        
        logger.info("✅ Agent运行完成")
        
    except Exception as e:
        logger.error(f"❌ 分析运行失败: {e}")
        if debug:
            import traceback
            traceback.print_exc()


async def run_comprehensive_analysis(settings: dict, symbol: str, output: str = None):
    """运行综合分析"""
    logger.info("🔄 启动综合分析工作流...")
    
    # 创建工作流
    workflow = ComprehensiveAnalysisWorkflow(settings)
    await workflow.initialize()
    
    # 运行分析
    analysis = await workflow.run_analysis(symbol)
    
    # 生成报告
    report = await generate_comprehensive_report(analysis)
    
    # 输出结果
    await output_results(report, "综合分析报告", output)
    
    # 显示关键指标
    logger.info(f"📊 综合评分: {analysis.overall_score}/100")
    logger.info(f"📈 市场方向: {analysis.market_outlook.direction.value}")
    logger.info(f"🎯 信号质量: {analysis.signal_quality:.1%}")
    logger.info(f"📋 数据完整性: {analysis.data_completeness:.1%}")
    
    if analysis.trading_signals:
        signal = analysis.trading_signals[0]
        logger.info(f"💡 交易信号: {signal.signal_type} - 强度{signal.strength:.1f}, 置信度{signal.confidence:.1%}")


async def run_single_agent_analysis(settings: dict, agent_class, agent_name: str, output: str = None, symbol: str = None):
    """运行单个Agent分析"""
    logger.info(f"🔄 启动{agent_name}...")
    
    # 创建Agent
    agent = agent_class(settings)
    await agent.initialize()
    
    # 运行分析
    if symbol and hasattr(agent, 'fetch_data') and 'symbol' in agent.fetch_data.__code__.co_varnames:
        # 期权Agent需要symbol参数
        data = await agent.fetch_data(symbol)
    else:
        data = await agent.fetch_data()
    
    analysis = await agent.analyze(data)
    report = await agent.generate_report(analysis)
    
    # 输出结果
    await output_results(report, agent_name, output)
    
    # 显示关键信息
    if hasattr(analysis, 'confidence'):
        logger.info(f"📊 分析置信度: {analysis.confidence:.1%}")
    if hasattr(analysis, 'policy_stance'):
        logger.info(f"📈 政策立场: {analysis.policy_stance.value}")


async def generate_comprehensive_report(analysis) -> str:
    """生成综合分析报告"""
    report = f"""
# 期货交易AI Agent - 综合分析报告

**分析时间**: {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**分析标的**: {analysis.target_symbol}
**分析版本**: {analysis.analysis_version}

## 📊 综合评估

- **综合评分**: {analysis.overall_score}/100
- **市场方向**: {analysis.market_outlook.direction.value}
- **信号质量**: {analysis.signal_quality:.1%}
- **数据完整性**: {analysis.data_completeness:.1%}
- **分析可靠性**: {analysis.analysis_reliability:.1%}

## 📈 市场展望

**方向**: {analysis.market_outlook.direction.value}
**置信度**: {analysis.market_outlook.confidence:.1%}
**时间范围**: {analysis.market_outlook.time_horizon.value}

**概率分析**:
- 上涨概率: {analysis.market_outlook.bullish_probability:.1%}
- 下跌概率: {analysis.market_outlook.bearish_probability:.1%}
- 横盘概率: {analysis.market_outlook.sideways_probability:.1%}

## 💡 交易信号

{generate_signals_section(analysis.trading_signals)}

## ⚠️ 风险评估

- **系统性风险**: {analysis.risk_assessment.systemic_risk.value if hasattr(analysis.risk_assessment.systemic_risk, 'value') else analysis.risk_assessment.systemic_risk}
- **流动性风险**: {analysis.risk_assessment.liquidity_risk}
- **波动性风险**: {analysis.risk_assessment.volatility_risk}

## 🎯 投资建议

{analysis.allocation_recommendation.equity_allocation:.1%} 股票配置
{analysis.allocation_recommendation.bond_allocation:.1%} 债券配置
{analysis.allocation_recommendation.cash_allocation:.1%} 现金配置

## 📋 执行摘要

{analysis.executive_summary}

## 🔍 关键要点

{chr(10).join(f"• {point}" for point in analysis.key_takeaways)}

## 📅 下次更新

**更新时间**: {analysis.next_update_time.strftime('%Y-%m-%d %H:%M:%S')}

---
*报告由期货交易AI Agent自动生成*
"""
    return report


def generate_signals_section(signals) -> str:
    """生成交易信号部分"""
    if not signals:
        return "当前无明确交易信号"
    
    sections = []
    for i, signal in enumerate(signals, 1):
        section = f"""
**信号 {i}**: {signal.signal_type}
- **强度**: {signal.strength:.1f}/1.0
- **置信度**: {signal.confidence:.1%}
- **时间范围**: {signal.time_horizon.value}
- **仓位大小**: {signal.position_size:.1%}
- **风险收益比**: {signal.risk_reward_ratio:.1f}
- **理由**: {signal.reasoning}
"""
        sections.append(section)
    
    return "\n".join(sections)


async def output_results(report: str, title: str, output_file: str = None):
    """输出分析结果"""
    
    # 打印到控制台
    logger.info(f"\n{'='*60}")
    logger.info(f"📋 {title}")
    logger.info(f"{'='*60}")
    print(report)
    
    # 保存到文件
    if output_file:
        output_path = Path(output_file)
    else:
        # 默认保存到reports目录
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{title.replace(' ', '_')}_{timestamp}.md"
        output_path = reports_dir / filename
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"📄 报告已保存到: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ 报告保存失败: {e}")


if __name__ == "__main__":
    main()
