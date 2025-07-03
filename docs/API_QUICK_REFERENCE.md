# API快速参考

## 核心Agent接口

### MacroPolicyAgent
```python
from agents.macro_policy_agent import MacroPolicyAgent

agent = MacroPolicyAgent(config)
await agent.initialize()
result = await agent.run()

# 结果结构
{
    'analysis': PolicyAnalysis(
        policy_stance: PolicyStance,
        confidence: float,
        sentiment_score: float,
        next_meeting_probability: Dict[str, float]
    ),
    'report': str
}
```

### GlobalSentimentAgent
```python
from agents.global_sentiment_agent import GlobalSentimentAgent

agent = GlobalSentimentAgent(config)
result = await agent.run()

# 结果结构
{
    'analysis': SentimentAnalysis(
        overall_sentiment: SentimentType,
        sentiment_score: float,
        fear_greed_index: int,
        risk_level: RiskLevel
    ),
    'report': str
}
```

### OptionStructureAgent
```python
from agents.option_structure_agent import OptionStructureAgent

agent = OptionStructureAgent(config)
result = await agent.run()

# 结果结构
{
    'analysis': OptionAnalysisResult(
        max_pain_level: float,
        gamma_exposure: float,
        quality_score: float
    ),
    'report': str
}
```

## 综合分析工作流

### ComprehensiveAnalysisWorkflow
```python
from workflows.comprehensive_analysis_workflow import ComprehensiveAnalysisWorkflow

workflow = ComprehensiveAnalysisWorkflow(config)
await workflow.initialize()
analysis = await workflow.run_analysis(target_symbol='SPY')

# 返回: ComprehensiveAnalysis
```

## 主要数据结构

### PolicyStance (枚举)
- `HAWKISH`: 鹰派
- `DOVISH`: 鸽派  
- `NEUTRAL`: 中性
- `MIXED`: 混合

### SentimentType (枚举)
- `POSITIVE`: 积极
- `NEGATIVE`: 消极
- `NEUTRAL`: 中性
- `MIXED`: 混合

### RiskLevel (枚举)
- `LOW`: 低风险
- `MEDIUM`: 中等风险
- `HIGH`: 高风险
- `EXTREME`: 极高风险

## 配置示例

```yaml
# config/settings.yaml
system:
  log_level: INFO
  timeout: 30

agents:
  macro_policy:
    enabled: true
  global_sentiment:
    enabled: true
  option_structure:
    enabled: true
```

```bash
# config/api_keys.env
OPENAI_API_KEY=your_key
FRED_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key
```

## 快速开始

```python
import asyncio
from workflows.comprehensive_analysis_workflow import ComprehensiveAnalysisWorkflow
from utils.config import load_config

async def main():
    config = load_config('config/settings.yaml')
    workflow = ComprehensiveAnalysisWorkflow(config)
    await workflow.initialize()
    
    analysis = await workflow.run_analysis('SPY')
    print(f"综合评分: {analysis.overall_score}/100")
    print(f"市场方向: {analysis.market_outlook.direction}")

asyncio.run(main())
``` 