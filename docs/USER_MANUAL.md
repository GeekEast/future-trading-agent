# 期货交易AI Agent - 用户手册

**版本**: v1.0.0  
**更新日期**: 2025-07-03  
**适用用户**: 交易员、分析师、投资者

---

## 🎯 快速开始

### 系统要求
- Python 3.10+
- 8GB+ RAM
- 稳定的网络连接

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-org/future-trading-agent.git
cd future-trading-agent
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置API密钥**
```bash
cp config/api_keys.env.example config/api_keys.env
# 编辑api_keys.env文件，填入你的API密钥
```

5. **运行测试**
```bash
python -m pytest tests/
```

---

## ⚙️ 配置指南

### API密钥配置

编辑 `config/api_keys.env` 文件：

```bash
# 必需的API密钥
OPENAI_API_KEY=sk-your-openai-key-here
FRED_API_KEY=your-32-character-fred-key
ALPHA_VANTAGE_API_KEY=your-16-character-av-key

# 可选的API密钥
POLYGON_API_KEY=your-polygon-key
QUANDL_API_KEY=your-quandl-key
```

### 获取API密钥

1. **OpenAI API** (必需)
   - 访问 https://platform.openai.com/
   - 创建账户并获取API密钥
   - 确保有足够的余额

2. **FRED API** (必需)
   - 访问 https://fred.stlouisfed.org/docs/api/
   - 免费注册并获取API密钥
   - 无使用限制

3. **Alpha Vantage API** (必需)
   - 访问 https://www.alphavantage.co/
   - 免费注册获取API密钥
   - 免费版有请求限制

### 系统配置

编辑 `config/settings.yaml` 文件：

```yaml
system:
  log_level: INFO      # 日志级别
  timeout: 30         # 超时时间(秒)
  max_retries: 3      # 最大重试次数

agents:
  macro_policy:
    enabled: true     # 是否启用宏观政策Agent
  global_sentiment:
    enabled: true     # 是否启用全球情绪Agent
  option_structure:
    enabled: true     # 是否启用期权结构Agent
```

---

## 🚀 基础使用

### 1. 单个Agent分析

#### 宏观政策分析

```python
import asyncio
from agents.macro_policy_agent import MacroPolicyAgent
from utils.config import load_config

async def macro_analysis():
    config = load_config('config/settings.yaml')
    agent = MacroPolicyAgent(config)
    
    await agent.initialize()
    result = await agent.run()
    
    print("=== 宏观政策分析 ===")
    analysis = result['analysis']
    print(f"政策立场: {analysis.policy_stance.value}")
    print(f"置信度: {analysis.confidence:.1%}")
    print(f"情绪分数: {analysis.sentiment_score:.2f}")
    print(f"\n报告:\n{result['report']}")

# 运行
asyncio.run(macro_analysis())
```

#### 全球情绪分析

```python
from agents.global_sentiment_agent import GlobalSentimentAgent

async def sentiment_analysis():
    config = load_config('config/settings.yaml')
    agent = GlobalSentimentAgent(config)
    
    await agent.initialize()
    result = await agent.run()
    
    print("=== 全球情绪分析 ===")
    analysis = result['analysis']
    print(f"整体情绪: {analysis.overall_sentiment.value}")
    print(f"情绪分数: {analysis.sentiment_score:.2f}")
    print(f"恐惧贪婪指数: {analysis.fear_greed_index}/100")
    print(f"风险级别: {analysis.risk_level.value}")

asyncio.run(sentiment_analysis())
```

#### 期权结构分析

```python
from agents.option_structure_agent import OptionStructureAgent

async def option_analysis():
    config = load_config('config/settings.yaml')
    agent = OptionStructureAgent(config)
    
    await agent.initialize()
    result = await agent.run()
    
    print("=== 期权结构分析 ===")
    analysis = result['analysis']
    print(f"最大痛点: ${analysis.max_pain_level:.2f}")
    print(f"Gamma敞口: {analysis.gamma_exposure:.2f}")
    print(f"质量评分: {analysis.quality_score:.0f}/100")

asyncio.run(option_analysis())
```

### 2. 综合分析

使用综合分析工作流获取完整的市场分析：

```python
from workflows.comprehensive_analysis_workflow import ComprehensiveAnalysisWorkflow

async def comprehensive_analysis():
    config = load_config('config/settings.yaml')
    workflow = ComprehensiveAnalysisWorkflow(config)
    
    await workflow.initialize()
    analysis = await workflow.run_analysis(target_symbol='SPY')
    
    print("=== 综合分析结果 ===")
    print(f"分析时间: {analysis.analysis_timestamp}")
    print(f"目标品种: {analysis.target_symbol}")
    print(f"综合评分: {analysis.overall_score:.0f}/100")
    print(f"信号质量: {analysis.signal_quality:.1%}")
    print(f"数据完整性: {analysis.data_completeness:.1%}")
    
    # 市场展望
    outlook = analysis.market_outlook
    print(f"\n市场方向: {outlook.direction.value}")
    print(f"置信度: {outlook.confidence:.1%}")
    print(f"看涨概率: {outlook.bullish_probability:.1%}")
    print(f"看跌概率: {outlook.bearish_probability:.1%}")
    
    # 交易信号
    print(f"\n交易信号:")
    for signal in analysis.trading_signals:
        print(f"- {signal.signal_type}: 强度{signal.strength:.1f}, 置信度{signal.confidence:.1%}")

asyncio.run(comprehensive_analysis())
```

---

## 📊 分析结果解读

### 宏观政策分析

#### 政策立场
- **HAWKISH (鹰派)**: 预期加息，关注通胀压力
- **DOVISH (鸽派)**: 预期降息，关注经济增长
- **NEUTRAL (中性)**: 政策保持稳定
- **MIXED (混合)**: 政策信号复杂

#### 置信度评级
- **90%+**: 极高置信度，强烈信号
- **80-90%**: 高置信度，可靠信号
- **70-80%**: 中等置信度，需要关注
- **<70%**: 低置信度，谨慎对待

#### 情绪分数
- **+0.5 到 +1.0**: 强烈鹰派
- **+0.1 到 +0.5**: 温和鹰派
- **-0.1 到 +0.1**: 中性
- **-0.5 到 -0.1**: 温和鸽派
- **-1.0 到 -0.5**: 强烈鸽派

### 全球情绪分析

#### 情绪类型
- **POSITIVE**: 市场乐观，风险偏好高
- **NEGATIVE**: 市场悲观，避险情绪浓
- **NEUTRAL**: 市场平衡，观望情绪
- **MIXED**: 情绪复杂，分歧较大

#### 恐惧贪婪指数
- **0-25**: 极度恐惧，可能超卖
- **25-45**: 恐惧，谨慎乐观
- **45-55**: 中性，平衡状态
- **55-75**: 贪婪，需要注意风险
- **75-100**: 极度贪婪，可能超买

#### 风险级别
- **LOW**: 低风险，市场相对稳定
- **MEDIUM**: 中等风险，需要关注
- **HIGH**: 高风险，谨慎操作
- **EXTREME**: 极高风险，避免重仓

### 期权结构分析

#### 最大痛点
- 期权到期时造成最大损失的价格水平
- 通常是支撑或阻力位
- 价格倾向于向最大痛点收敛

#### Gamma敞口
- **正Gamma**: 做市商买入保护，推高波动
- **负Gamma**: 做市商卖出对冲，压制波动
- **零Gamma**: 平衡点，关键技术位

#### 质量评分
- **80-100**: 优秀，数据完整，信号可靠
- **60-80**: 良好，数据基本完整
- **40-60**: 一般，数据有缺失
- **<40**: 较差，数据不完整

---

## 🎯 交易策略应用

### 基于宏观分析的策略

#### 鹰派环境策略
```
- 关注利率敏感板块的做空机会
- 考虑做空长期债券
- 关注美元走强对商品的影响
- 减少成长股权重
```

#### 鸽派环境策略
```
- 增加成长股配置
- 考虑做多长期债券
- 关注流动性驱动的资产
- 增加风险资产权重
```

### 基于情绪分析的策略

#### 极度恐惧时
```
- 逢低买入优质资产
- 关注超卖反弹机会
- 增加防御性配置
- 等待情绪修复
```

#### 极度贪婪时
```
- 考虑获利了结
- 增加对冲保护
- 降低风险敞口
- 关注反转信号
```

### 基于期权结构的策略

#### 负Gamma环境
```
- 预期低波动率
- 适合卖出波动率策略
- 关注区间交易
- 避免追涨杀跌
```

#### 正Gamma环境
```
- 预期高波动率
- 适合买入波动率策略
- 关注突破交易
- 准备快速反应
```

---

## 🔧 高级功能

### 自定义分析周期

```python
from datetime import datetime, timedelta

# 自定义数据获取周期
config = {
    'fred_api_key': 'your_key',
    'data_period': {
        'start_date': datetime.now() - timedelta(days=365),
        'end_date': datetime.now()
    }
}
```

### 批量分析

```python
async def batch_analysis():
    symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
    workflow = ComprehensiveAnalysisWorkflow(config)
    await workflow.initialize()
    
    results = {}
    for symbol in symbols:
        results[symbol] = await workflow.run_analysis(symbol)
    
    return results
```

### 定时分析

```python
import schedule
import time

def scheduled_analysis():
    # 每天早上8点运行分析
    schedule.every().day.at("08:00").do(run_analysis)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

def run_analysis():
    asyncio.run(comprehensive_analysis())
```

---

## 📈 实战案例

### 案例1：FOMC会议前分析

```python
async def fomc_analysis():
    """FOMC会议前的综合分析"""
    config = load_config('config/settings.yaml')
    
    # 重点关注宏观政策Agent
    macro_agent = MacroPolicyAgent(config)
    await macro_agent.initialize()
    
    result = await macro_agent.run()
    analysis = result['analysis']
    
    print("=== FOMC会议前分析 ===")
    print(f"当前政策立场: {analysis.policy_stance.value}")
    print(f"分析置信度: {analysis.confidence:.1%}")
    
    # 下次会议概率
    print("下次会议预测:")
    for action, prob in analysis.next_meeting_probability.items():
        print(f"  {action}: {prob:.1%}")
    
    # 重点关注指标
    print("\n关键指标:")
    for indicator in analysis.key_indicators:
        print(f"  - {indicator}")
    
    # 风险提示
    print("\n风险因素:")
    for risk in analysis.risk_factors:
        print(f"  - {risk}")

asyncio.run(fomc_analysis())
```

### 案例2：期权到期日分析

```python
async def opex_analysis():
    """期权到期日分析"""
    config = load_config('config/settings.yaml')
    
    option_agent = OptionStructureAgent(config)
    await option_agent.initialize()
    
    result = await option_agent.run()
    analysis = result['analysis']
    
    print("=== 期权到期日分析 ===")
    print(f"最大痛点: ${analysis.max_pain_level:.2f}")
    print(f"当前价格距离痛点: {((current_price/analysis.max_pain_level-1)*100):.1f}%")
    
    # Gamma敞口影响
    if analysis.gamma_exposure > 0:
        print("正Gamma环境：预期波动率上升")
    else:
        print("负Gamma环境：预期波动率下降")
    
    # 关键价位
    print(f"\n支撑位: {analysis.support_levels}")
    print(f"阻力位: {analysis.resistance_levels}")

asyncio.run(opex_analysis())
```

### 案例3：市场风险评估

```python
async def risk_assessment():
    """市场风险评估"""
    config = load_config('config/settings.yaml')
    workflow = ComprehensiveAnalysisWorkflow(config)
    await workflow.initialize()
    
    analysis = await workflow.run_analysis('SPY')
    risk = analysis.risk_assessment
    
    print("=== 市场风险评估 ===")
    print(f"整体风险级别: {risk.overall_risk_level.value}")
    print(f"风险评分: {risk.risk_score:.1f}/100")
    
    # 各维度风险
    print("\n风险分解:")
    print(f"  宏观风险: {risk.macro_risk:.1f}")
    print(f"  情绪风险: {risk.sentiment_risk:.1f}")
    print(f"  技术风险: {risk.technical_risk:.1f}")
    
    # 风险建议
    print("\n风险建议:")
    for suggestion in risk.risk_mitigation_suggestions:
        print(f"  - {suggestion}")

asyncio.run(risk_assessment())
```

---

## ⚠️ 注意事项

### 数据使用限制

1. **API限制**
   - Alpha Vantage: 免费版5次/分钟
   - FRED: 无限制
   - 注意API额度管理

2. **数据延迟**
   - 宏观数据可能有几天延迟
   - 期权数据实时性较好
   - 新闻数据基本实时

3. **市场时间**
   - 考虑交易时间和时区
   - 节假日数据可能不可用
   - 盘前盘后数据质量差异

### 风险提示

1. **投资风险**
   - 本系统仅提供分析参考
   - 投资决策需自行承担风险
   - 建议结合其他分析工具

2. **技术风险**
   - 数据源可能不可用
   - 模型可能存在偏差
   - 系统可能出现故障

3. **使用建议**
   - 定期更新API密钥
   - 监控系统运行状态
   - 建立数据备份机制

---

## 🛠️ 故障排除

### 常见问题

1. **API密钥错误**
```bash
Error: Invalid API key
Solution: 检查config/api_keys.env文件中的密钥是否正确
```

2. **网络连接问题**
```bash
Error: Connection timeout
Solution: 检查网络连接，增加timeout设置
```

3. **数据获取失败**
```bash
Error: No data available
Solution: 检查数据源状态，确认市场开放时间
```

### 日志查看

```bash
# 查看实时日志
tail -f logs/agent.log

# 查看错误日志
grep ERROR logs/agent.log
```

### 调试模式

```python
# 启用调试模式
import logging
logging.basicConfig(level=logging.DEBUG)

# 或修改config/settings.yaml
system:
  log_level: DEBUG
```

---

## 📞 技术支持

### 获取帮助

1. **文档**: 查看完整的API文档
2. **示例**: 参考examples/目录下的示例代码
3. **社区**: 加入技术讨论群
4. **邮件**: james@sapia.ai

### 反馈建议

我们欢迎您的反馈和建议：
- 功能需求
- 性能优化
- 错误报告
- 使用体验

---

**免责声明**: 本系统提供的分析结果仅供参考，不构成投资建议。用户应根据自身情况做出投资决策，并自行承担投资风险。 