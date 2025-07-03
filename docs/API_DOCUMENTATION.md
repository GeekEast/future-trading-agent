# 期货交易AI Agent - API文档

**版本**: v1.0.0  
**更新日期**: 2025-07-03  
**适用范围**: 开发者、集成商、技术用户

---

## 📋 目录

1. [系统概述](#系统概述)
2. [Agent接口](#agent接口)
3. [数据模型](#数据模型)
4. [配置管理](#配置管理)
5. [工作流API](#工作流api)
6. [错误处理](#错误处理)
7. [使用示例](#使用示例)

---

## 🎯 系统概述

### 架构设计
期货交易AI Agent系统采用基于LangGraph的多Agent协同架构，包含三个核心Agent：

- **MacroPolicy Agent**: 宏观政策分析
- **GlobalSentiment Agent**: 全球情绪分析  
- **OptionStructure Agent**: 期权结构分析

### 核心特性
- **异步处理**: 所有API调用支持异步操作
- **类型安全**: 完整的类型提示支持
- **错误处理**: 完善的异常处理机制
- **可扩展**: 模块化设计，支持自定义Agent

---

## 🤖 Agent接口

### BaseAgent抽象类

所有Agent继承自BaseAgent抽象类，提供统一的接口：

```python
class BaseAgent(ABC):
    """Agent基础抽象类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化Agent
        
        Args:
            name: Agent名称
            config: 配置字典
        """
        
    @abstractmethod
    async def initialize(self) -> None:
        """初始化Agent"""
        
    @abstractmethod
    async def fetch_data(self) -> Dict[str, Any]:
        """获取数据"""
        
    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析数据"""
        
    @abstractmethod
    async def generate_report(self, analysis: Dict[str, Any]) -> str:
        """生成报告"""
        
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """获取依赖的其他Agent"""
        
    async def run(self) -> Dict[str, Any]:
        """运行Agent完整流程"""
```

### MacroPolicy Agent

#### 初始化

```python
from agents.macro_policy_agent import MacroPolicyAgent

# 配置
config = {
    'fred_api_key': 'YOUR_FRED_API_KEY',
    'timeout': 30
}

# 创建Agent
agent = MacroPolicyAgent(config)
await agent.initialize()
```

#### 核心方法

```python
# 获取宏观经济数据
data = await agent.fetch_data()
# 返回格式:
{
    'source': 'FRED',
    'timestamp': datetime,
    'indicators': List[EconomicIndicator],
    'total_records': int
}

# 分析政策环境
analysis = await agent.analyze(data)
# 返回类型: PolicyAnalysis

# 生成报告
report = await agent.generate_report(analysis)
# 返回类型: str
```

#### PolicyAnalysis结果

```python
@dataclass
class PolicyAnalysis:
    """政策分析结果"""
    policy_stance: PolicyStance          # 政策立场
    confidence: float                    # 置信度 (0-1)
    key_indicators: List[str]            # 关键指标
    risk_factors: List[str]              # 风险因素
    recommendations: List[str]           # 建议
    next_meeting_probability: Dict[str, float]  # 下次会议概率
    sentiment_score: float               # 情绪分数 (-1到1)
```

### GlobalSentiment Agent

#### 初始化

```python
from agents.global_sentiment_agent import GlobalSentimentAgent

config = {
    'alpha_vantage_api_key': 'YOUR_ALPHA_VANTAGE_KEY',
    'timeout': 30
}

agent = GlobalSentimentAgent(config)
await agent.initialize()
```

#### 核心方法

```python
# 获取全球情绪数据
data = await agent.fetch_data()
# 返回格式:
{
    'global_indices': List[GlobalIndex],
    'volatility_data': List[VolatilityIndicator],
    'news_sentiment': List[NewsEventSentiment],
    'cross_asset_correlation': Dict[str, float]
}

# 分析市场情绪
analysis = await agent.analyze(data)
# 返回类型: SentimentAnalysis
```

#### SentimentAnalysis结果

```python
@dataclass
class SentimentAnalysis:
    """情绪分析结果"""
    overall_sentiment: SentimentType     # 整体情绪
    sentiment_score: float               # 情绪分数 (-1到1)
    confidence: float                    # 置信度
    fear_greed_index: int                # 恐惧贪婪指数 (0-100)
    risk_level: RiskLevel                # 风险级别
    key_drivers: List[str]               # 关键驱动因素
    market_regime: str                   # 市场状态
    volatility_assessment: Dict[str, Any] # 波动率评估
```

### OptionStructure Agent

#### 初始化

```python
from agents.option_structure_agent import OptionStructureAgent

config = {
    'yahoo_finance_timeout': 30
}

agent = OptionStructureAgent(config)
await agent.initialize()
```

#### 核心方法

```python
# 获取期权数据
data = await agent.fetch_data(symbol='SPY')
# 返回格式:
{
    'symbol': str,
    'current_price': float,
    'options_chain': List[OptionContract],
    'expiration_dates': List[datetime]
}

# 分析期权结构
analysis = await agent.analyze(data)
# 返回类型: OptionAnalysisResult
```

#### OptionAnalysisResult结果

```python
@dataclass
class OptionAnalysisResult:
    """期权分析结果"""
    max_pain_level: float                # 最大痛点
    gamma_exposure: float                # Gamma敞口
    zero_gamma_level: float              # 零Gamma水平
    call_wall: float                     # Call墙
    put_wall: float                      # Put墙
    implied_volatility_rank: float       # 隐含波动率排名
    support_levels: List[float]          # 支撑位
    resistance_levels: List[float]       # 阻力位
    expected_move: float                 # 预期波动
    quality_score: float                 # 质量评分
```

---

## 📊 数据模型

### 基础数据类型

#### EconomicIndicator

```python
@dataclass
class EconomicIndicator:
    """经济指标数据"""
    symbol: str                          # 指标代码
    name: str                            # 指标名称
    value: float                         # 指标值
    unit: str                            # 单位
    release_date: datetime               # 发布日期
    indicator_type: EconomicIndicatorType # 指标类型
    source: str                          # 数据源
```

#### MarketDataPoint

```python
@dataclass
class MarketDataPoint:
    """市场数据点"""
    symbol: str                          # 交易品种
    timestamp: datetime                  # 时间戳
    price: float                         # 价格
    volume: Optional[int]                # 成交量
    bid: Optional[float]                 # 买价
    ask: Optional[float]                 # 卖价
    source: str                          # 数据源
```

#### OptionContract

```python
@dataclass
class OptionContract:
    """期权合约"""
    symbol: str                          # 期权代码
    underlying_symbol: str               # 标的代码
    strike: float                        # 行权价
    expiration: datetime                 # 到期日
    option_type: OptionType              # 期权类型 (CALL/PUT)
    last_price: float                    # 最新价格
    bid: float                           # 买价
    ask: float                           # 卖价
    volume: int                          # 成交量
    open_interest: int                   # 持仓量
    implied_volatility: Optional[float]  # 隐含波动率
```

### 分析结果类型

#### ComprehensiveAnalysis

```python
@dataclass
class ComprehensiveAnalysis:
    """综合分析结果"""
    analysis_timestamp: datetime         # 分析时间
    target_symbol: str                   # 目标品种
    macro_analysis: Optional[PolicyAnalysis]      # 宏观分析
    sentiment_analysis: Optional[SentimentAnalysis] # 情绪分析
    option_analysis: Optional[OptionAnalysisResult] # 期权分析
    consensus: AgentConsensus            # Agent共识
    market_outlook: MarketOutlook        # 市场展望
    trading_signals: List[TradingSignal] # 交易信号
    risk_assessment: RiskAssessment      # 风险评估
    overall_score: float                 # 综合评分
    signal_quality: float                # 信号质量
    data_completeness: float             # 数据完整性
```

---

## ⚙️ 配置管理

### 配置文件结构

```yaml
# config/settings.yaml
system:
  log_level: INFO
  timeout: 30
  max_retries: 3

data_sources:
  fred:
    base_url: "https://api.stlouisfed.org/fred"
    timeout: 30
    
  alpha_vantage:
    base_url: "https://www.alphavantage.co"
    timeout: 30
    
  yahoo_finance:
    timeout: 30

agents:
  macro_policy:
    indicator_weights:
      FEDFUNDS: 0.25
      UNRATE: 0.20
      CPIAUCSL: 0.20
      GDP: 0.15
      DGS10: 0.10
      PAYEMS: 0.10
      
  global_sentiment:
    sentiment_weights:
      news: 0.4
      volatility: 0.3
      correlation: 0.3
      
  option_structure:
    analysis_parameters:
      min_volume: 10
      min_open_interest: 100
```

### 配置加载

```python
from utils.config import load_config

# 加载配置
config = load_config('config/settings.yaml')

# 访问配置
fred_config = config['data_sources']['fred']
agent_weights = config['agents']['macro_policy']['indicator_weights']
```

### 环境变量配置

```bash
# config/api_keys.env
OPENAI_API_KEY=your_openai_key
FRED_API_KEY=your_fred_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
POLYGON_API_KEY=your_polygon_key
QUANDL_API_KEY=your_quandl_key
```

---

## 🔄 工作流API

### ComprehensiveAnalysisWorkflow

综合分析工作流协调所有Agent并生成最终报告：

```python
from workflows.comprehensive_analysis_workflow import ComprehensiveAnalysisWorkflow

# 初始化工作流
workflow = ComprehensiveAnalysisWorkflow(config)
await workflow.initialize()

# 运行综合分析
analysis = await workflow.run_analysis(target_symbol='SPY')
```

### 工作流状态

```python
@dataclass
class AnalysisState:
    """分析状态"""
    target_symbol: str                   # 目标品种
    analysis_timestamp: datetime         # 分析时间
    config: Dict[str, Any]               # 配置
    
    # Agent分析结果
    macro_analysis: Optional[PolicyAnalysis]
    sentiment_analysis: Optional[SentimentAnalysis]
    option_analysis: Optional[OptionAnalysisResult]
    
    # 最终结果
    comprehensive_analysis: Optional[ComprehensiveAnalysis]
    final_report: Optional[str]
    
    # 状态控制
    current_step: str
    completed_agents: List[str]
    failed_agents: List[str]
```

---

## ⚠️ 错误处理

### 异常类型

```python
class AgentError(Exception):
    """Agent基础异常"""
    
class DataSourceError(AgentError):
    """数据源异常"""
    
class AnalysisError(AgentError):
    """分析异常"""
    
class ConfigurationError(AgentError):
    """配置异常"""
```

### 错误处理示例

```python
try:
    # 运行Agent
    result = await agent.run()
except DataSourceError as e:
    logger.error(f"数据源错误: {e}")
    # 处理数据源错误
except AnalysisError as e:
    logger.error(f"分析错误: {e}")
    # 处理分析错误
except Exception as e:
    logger.error(f"未知错误: {e}")
    # 处理其他错误
```

---

## 📝 使用示例

### 基础使用

```python
import asyncio
from agents.macro_policy_agent import MacroPolicyAgent
from utils.config import load_config

async def main():
    # 加载配置
    config = load_config('config/settings.yaml')
    
    # 创建Agent
    agent = MacroPolicyAgent(config)
    
    # 初始化
    await agent.initialize()
    
    # 运行分析
    result = await agent.run()
    
    # 获取结果
    analysis = result['analysis']
    report = result['report']
    
    print(f"政策立场: {analysis.policy_stance}")
    print(f"置信度: {analysis.confidence:.2%}")
    print(f"报告:\n{report}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 综合分析示例

```python
import asyncio
from workflows.comprehensive_analysis_workflow import ComprehensiveAnalysisWorkflow
from utils.config import load_config

async def comprehensive_analysis():
    # 加载配置
    config = load_config('config/settings.yaml')
    
    # 创建工作流
    workflow = ComprehensiveAnalysisWorkflow(config)
    await workflow.initialize()
    
    # 运行综合分析
    analysis = await workflow.run_analysis(target_symbol='SPY')
    
    # 输出结果
    print(f"综合评分: {analysis.overall_score}/100")
    print(f"市场展望: {analysis.market_outlook.direction}")
    print(f"信号质量: {analysis.signal_quality:.2%}")
    
    # 交易信号
    for signal in analysis.trading_signals:
        print(f"交易信号: {signal.signal_type} - 强度: {signal.strength:.2f}")

if __name__ == "__main__":
    asyncio.run(comprehensive_analysis())
```

### 自定义Agent开发

```python
from agents.base_agent import BaseAgent
from typing import Dict, Any, List

class CustomAgent(BaseAgent):
    """自定义Agent示例"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("CustomAgent", config)
        
    async def initialize(self) -> None:
        """初始化自定义Agent"""
        await super().initialize()
        # 添加自定义初始化逻辑
        
    async def fetch_data(self) -> Dict[str, Any]:
        """获取自定义数据"""
        # 实现数据获取逻辑
        return {
            'custom_data': 'your_data_here',
            'timestamp': datetime.now()
        }
        
    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析自定义数据"""
        # 实现分析逻辑
        return {
            'analysis_result': 'your_analysis_here',
            'confidence': 0.85
        }
        
    async def generate_report(self, analysis: Dict[str, Any]) -> str:
        """生成自定义报告"""
        return f"自定义分析报告: {analysis['analysis_result']}"
        
    def get_dependencies(self) -> List[str]:
        """获取依赖"""
        return []  # 无依赖
```

---

## 🔍 调试和监控

### 日志配置

```python
from utils.logger import setup_logger

# 设置日志
logger = setup_logger(
    level="INFO",
    format="<green>{time}</green> | <level>{level}</level> | <cyan>{name}</cyan> | {message}",
    rotation="1 day",
    retention="30 days"
)
```

### 性能监控

```python
import time
from functools import wraps

def performance_monitor(func):
    """性能监控装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"{func.__name__} 执行时间: {end_time - start_time:.2f}秒")
        return result
    return wrapper

# 使用示例
@performance_monitor
async def monitored_function():
    # 你的函数逻辑
    pass
```

---

## 📞 技术支持

### 联系信息
- **技术支持**: james@sapia.ai
- **文档更新**: 2025-07-03
- **API版本**: v1.0.0

### 常见问题
1. **Q: 如何处理API密钥？**
   A: 将API密钥存储在环境变量中，不要硬编码在代码中。

2. **Q: 如何扩展新的数据源？**
   A: 继承BaseDataSource类，实现所需的方法。

3. **Q: 如何优化性能？**
   A: 使用异步编程，合理设置超时时间，实现数据缓存。

---

**免责声明**: 本API文档提供的信息仅供参考，实际使用时请结合具体业务需求进行调整。 