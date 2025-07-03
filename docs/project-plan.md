# Project Plan


继续框架:
- python
- langchain + langgraph
- run in command

✅ 整合后的盘前分析 AI Agent 架构
🔹 Agent 1：宏观 + 美联储监控 Agent（简称：MacroPolicy Agent）

📌 职责：
提取过去24-48小时内全球重要宏观经济数据（就业、通胀、GDP、PMI、零售销售等）

追踪最新的美联储动态，包括：
- FOMC 官员讲话
- 利率点阵图、FedWatch 利率预期
- 美联储资产负债表变动、逆回购与贴现窗口流动性变化

📊 输出内容：
- 当日对期货交易有潜在影响的宏观事件摘要
- 利率预期变化图表 + 利率期货定价
- 市场解读与偏离分析（例如 CPI 超预期 → 利空股指）

🔗 推荐数据源/API：
- TradingEconomics API
- CME FedWatch Tool
- St. Louis FRED API
- FRB Events / Press

🔹 Agent 2：全球市场与事件风险 Agent（简称：GlobalSentiment Agent）

📌 职责：
汇总前夜全球主要市场表现：
- 美股三大指数期货
- 欧股、亚太股市
- 美债收益率曲线
- 美元指数 DXY
- 原油、黄金、铜、比特币等大宗商品
- 抽取当日重要宏观日程与事件风险（财经日历）
- 经济数据发布时间表
- 财报公布（大型公司）
- 地缘政治或突发新闻

📊 输出内容：
- 全球资产表现概览（红绿变动列表 +突出异常行情）
- 今日高影响日程清单（含预期值与前值）
- 风险事件评估打分（低/中/高）

🔗 推荐数据源/API：
- Yahoo Finance API (unofficial)
- Investing.com Calendar
- Tiingo
- Alpha Vantage
- NewsAPI（可用于提取头条新闻标题）

🔹 Agent 3：期权市场结构分析 Agent（简称：OptionStructure Agent）

📌 职责：
分析与 ES/NQ/RTY/GC 对应的 ETF 或指数（SPX/SPY、QQQ、IWM、GLD）的期权市场结构：
- 未平仓量分布（OI）
- 最大痛点位（Max Pain）
- 支撑/阻力（Put/Call 墙）
- 当日 Gamma 区间（正负 Gamma 转折位）

提取整体市场情绪与波动指标：
- VIX、VVIX、MOVE
- 隐含波动率曲面（IV Skew）
- Put/Call Ratio（情绪倾向）

📊 输出内容：
- 支撑/阻力图 + 热力图分布
- Gamma 曝险与流动性预警
- 波动率预期图表（含 VIX term structure）
- 综合市场情绪评分（Fear/Neutral/Greed）

🔗 推荐数据源/API：
- SpotGamma（高级推荐）
- OptionStrat
- Barchart Options API
- CBOE Data
- Market Chameleon

📋 输出示例结构（每日盘前 Dashboard）

📈 MacroPolicy Summary
- 美国上周初请失业金人数低于预期（18.7万 vs 预期 22万）→ 利多
- FedWatch 显示 9月降息概率下滑至 38%
- FOMC 官员 Daly 昨日讲话仍偏鹰派

🌐 GlobalSentiment Snapshot
- 纳指期货 +0.6%，日经225 -0.3%，10Y美债收益率上行至4.31%
- 今日 20:30 美东时间公布非农就业数据（预期18万，前值20.5万）
- OPEC 将于今日召开紧急会议 → 油价 +2.1%

🧮 OptionStructure Metrics
- SPX Call墙：5450，Put墙：5350，最大痛点：5400
- 当日正负Gamma区间：5385 - 5430，偏向震荡
- VIX：14.2（平稳），P/C Ratio：0.62（偏多情绪）

