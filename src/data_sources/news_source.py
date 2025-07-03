"""
新闻数据源 - 通过RSS和免费API获取财经新闻
支持情绪分析和地缘政治事件监控
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import feedparser
import httpx
from bs4 import BeautifulSoup

from .base_source import BaseDataSource
from models.sentiment_data import NewsEventSentiment, GeopoliticalEvent, RiskLevel
from utils.helpers import convert_to_numeric


class NewsDataSource(BaseDataSource):
    """新闻数据源 - 基于RSS和免费API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("NewsSource", config)
        
        # 主要财经新闻RSS源（使用更可靠的源）
        self.rss_feeds = {
            'bbc_business': 'https://feeds.bbci.co.uk/news/business/rss.xml',
            'cnn_business': 'http://rss.cnn.com/rss/money_latest.rss',
            'marketwatch': 'https://feeds.content.dowjones.io/public/rss/mw_marketpulse'
        }
        
        # 地缘政治新闻源
        self.geopolitical_feeds = {
            'bbc_world': 'https://feeds.bbci.co.uk/news/world/rss.xml',
            'cnn_world': 'http://rss.cnn.com/rss/edition_world.rss'
        }
        
        # 情绪分析关键词
        self.sentiment_keywords = {
            'positive': [
                'rally', 'surge', 'gains', 'optimism', 'bullish', 'confident',
                'growth', 'recovery', 'boost', 'strong', 'positive', 'upgrade',
                'breakthrough', 'success', 'agreement', 'deal', 'progress'
            ],
            'negative': [
                'crash', 'plunge', 'decline', 'pessimism', 'bearish', 'concern',
                'recession', 'crisis', 'risk', 'weak', 'negative', 'downgrade',
                'conflict', 'tension', 'uncertainty', 'fear', 'panic', 'sell-off'
            ],
            'neutral': [
                'stable', 'unchanged', 'steady', 'monitor', 'watch', 'expect',
                'maintain', 'continue', 'ongoing', 'regular', 'normal'
            ]
        }
        
        # 市场相关性关键词
        self.market_keywords = [
            'fed', 'federal reserve', 'interest rate', 'inflation', 'gdp',
            'unemployment', 'dollar', 'oil', 'gold', 'stock market', 'bond',
            'treasury', 'yield', 'volatility', 'vix', 'earnings', 'trade'
        ]
        
        # 地缘政治关键词
        self.geopolitical_keywords = {
            'high_risk': ['war', 'conflict', 'sanctions', 'military', 'attack'],
            'medium_risk': ['tension', 'dispute', 'protest', 'election', 'policy'],
            'low_risk': ['cooperation', 'partnership', 'agreement', 'negotiation']
        }
        
    async def _custom_initialize(self) -> None:
        """初始化新闻数据源"""
        self.logger.info("新闻数据源初始化完成")
        
    async def fetch_data(
        self, 
        sources: List[str] = None, 
        hours_back: int = 24,
        max_articles: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """获取新闻数据"""
        
        if sources is None:
            sources = list(self.rss_feeds.keys())
            
        news_data = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # 获取财经新闻
        for source in sources:
            if source in self.rss_feeds:
                articles = await self._fetch_rss_news(
                    self.rss_feeds[source], source, cutoff_time
                )
                news_data.extend(articles)
                
        # 获取地缘政治新闻
        geo_sources = kwargs.get('include_geopolitical', True)
        if geo_sources:
            for source, url in self.geopolitical_feeds.items():
                articles = await self._fetch_rss_news(url, source, cutoff_time)
                news_data.extend(articles)
                
        # 限制文章数量
        news_data = news_data[:max_articles]
        
        # 分析情绪
        analyzed_news = []
        for article in news_data:
            sentiment = self._analyze_sentiment(article)
            analyzed_news.append(sentiment)
            
        return {
            'source': 'NewsSource',
            'timestamp': datetime.now(),
            'articles_count': len(analyzed_news),
            'news_sentiment': analyzed_news,
            'metadata': {
                'sources_used': sources,
                'hours_back': hours_back,
                'cutoff_time': cutoff_time.isoformat()
            }
        }
        
    async def _fetch_rss_news(
        self, 
        feed_url: str, 
        source: str, 
        cutoff_time: datetime
    ) -> List[Dict[str, Any]]:
        """从RSS源获取新闻"""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(feed_url)
                response.raise_for_status()
                
            feed = feedparser.parse(response.content)
            articles = []
            
            for entry in feed.entries:
                try:
                    # 解析发布时间
                    published_time = self._parse_publish_time(entry)
                    if published_time < cutoff_time:
                        continue
                        
                    # 提取文章内容
                    title = entry.get('title', '')
                    summary = entry.get('summary', '') or entry.get('description', '')
                    link = entry.get('link', '')
                    
                    # 清理HTML标签
                    content = self._clean_html(summary)
                    
                    article = {
                        'headline': title,
                        'content': content,
                        'url': link,
                        'source': source,
                        'published_time': published_time
                    }
                    
                    articles.append(article)
                    
                except Exception as e:
                    self.logger.warning(f"解析文章失败 {source}: {e}")
                    continue
                    
            return articles
            
        except Exception as e:
            self.logger.error(f"获取RSS新闻失败 {source}: {e}")
            return []
            
    def _parse_publish_time(self, entry) -> datetime:
        """解析发布时间"""
        try:
            # 尝试不同的时间字段
            time_str = entry.get('published') or entry.get('updated') or entry.get('pubDate')
            
            if time_str:
                # 使用feedparser的时间解析
                time_struct = entry.get('published_parsed') or entry.get('updated_parsed')
                if time_struct:
                    return datetime(*time_struct[:6])
                    
            # 如果无法解析，返回当前时间
            return datetime.now()
            
        except Exception:
            return datetime.now()
            
    def _clean_html(self, text: str) -> str:
        """清理HTML标签"""
        try:
            if not text:
                return ""
                
            # 使用BeautifulSoup清理HTML
            soup = BeautifulSoup(text, 'html.parser')
            clean_text = soup.get_text()
            
            # 清理多余空白
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            return clean_text[:1000]  # 限制长度
            
        except Exception:
            return text[:1000] if text else ""
            
    def _analyze_sentiment(self, article: Dict[str, Any]) -> NewsEventSentiment:
        """分析文章情绪"""
        title = article.get('headline', '').lower()
        content = article.get('content', '').lower()
        full_text = f"{title} {content}"
        
        # 计算情绪分数
        positive_score = sum(1 for word in self.sentiment_keywords['positive'] if word in full_text)
        negative_score = sum(1 for word in self.sentiment_keywords['negative'] if word in full_text)
        neutral_score = sum(1 for word in self.sentiment_keywords['neutral'] if word in full_text)
        
        total_sentiment_words = positive_score + negative_score + neutral_score
        
        if total_sentiment_words == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (positive_score - negative_score) / max(total_sentiment_words, 1)
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
        # 计算市场相关性
        market_relevance = sum(1 for keyword in self.market_keywords if keyword in full_text)
        market_relevance = min(1.0, market_relevance / 5.0)  # 归一化到0-1
        
        # 提取关键词
        keywords = []
        for keyword in self.market_keywords:
            if keyword in full_text:
                keywords.append(keyword)
                
        # 识别受影响的金融工具
        affected_instruments = self._identify_affected_instruments(full_text)
        
        # 计算重要性
        importance = self._calculate_importance(article, market_relevance, total_sentiment_words)
        
        return NewsEventSentiment(
            headline=article.get('headline', ''),
            content=article.get('content', ''),
            source=article.get('source', ''),
            sentiment_score=sentiment_score,
            importance=importance,
            market_relevance=market_relevance,
            affected_instruments=affected_instruments,
            keywords=keywords,
            published_time=article.get('published_time', datetime.now()),
            analyzed_time=datetime.now()
        )
        
    def _identify_affected_instruments(self, text: str) -> List[str]:
        """识别受影响的金融工具"""
        instruments = []
        
        # 检查主要资产类别
        if any(word in text for word in ['stock', 'equity', 'share', 's&p', 'nasdaq', 'dow']):
            instruments.append('equities')
            
        if any(word in text for word in ['bond', 'treasury', 'yield', 'debt']):
            instruments.append('bonds')
            
        if any(word in text for word in ['dollar', 'currency', 'forex', 'exchange rate']):
            instruments.append('currencies')
            
        if any(word in text for word in ['oil', 'gold', 'commodity', 'crude']):
            instruments.append('commodities')
            
        if any(word in text for word in ['bitcoin', 'crypto', 'digital currency']):
            instruments.append('crypto')
            
        return instruments
        
    def _calculate_importance(
        self, 
        article: Dict[str, Any], 
        market_relevance: float, 
        sentiment_words: int
    ) -> float:
        """计算文章重要性"""
        importance = 0.0
        
        # 基于市场相关性
        importance += market_relevance * 0.4
        
        # 基于情绪词数量
        importance += min(sentiment_words / 10.0, 0.3)
        
        # 基于来源权重
        source_weights = {
            'reuters': 0.3,
            'bloomberg': 0.3,
            'wsj': 0.25,
            'ft': 0.2,
            'cnbc': 0.15,
            'marketwatch': 0.1
        }
        
        source = article.get('source', '').lower()
        for key, weight in source_weights.items():
            if key in source:
                importance += weight
                break
        else:
            importance += 0.05  # 默认权重
            
        return min(1.0, importance)
        
    async def get_geopolitical_events(self, hours_back: int = 48) -> List[GeopoliticalEvent]:
        """获取地缘政治事件"""
        news_data = await self.fetch_data(
            sources=list(self.geopolitical_feeds.keys()),
            hours_back=hours_back,
            include_geopolitical=True
        )
        
        events = []
        for article in news_data.get('news_sentiment', []):
            event = self._extract_geopolitical_event(article)
            if event:
                events.append(event)
                
        return events
        
    def _extract_geopolitical_event(self, article: NewsEventSentiment) -> Optional[GeopoliticalEvent]:
        """从新闻中提取地缘政治事件"""
        full_text = f"{article.headline} {article.content}".lower()
        
        # 检查是否包含地缘政治关键词
        risk_level = RiskLevel.LOW
        severity_score = 0
        
        for risk, keywords in self.geopolitical_keywords.items():
            for keyword in keywords:
                if keyword in full_text:
                    if risk == 'high_risk':
                        severity_score += 3
                    elif risk == 'medium_risk':
                        severity_score += 2
                    else:
                        severity_score += 1
                        
        if severity_score == 0:
            return None
            
        # 确定风险级别
        if severity_score >= 6:
            risk_level = RiskLevel.EXTREME
        elif severity_score >= 4:
            risk_level = RiskLevel.HIGH
        elif severity_score >= 2:
            risk_level = RiskLevel.MEDIUM
            
        # 尝试提取国家/地区
        country = self._extract_country(full_text)
        region = self._get_region_from_country(country)
        
        # 生成事件ID
        event_id = f"geo_{article.published_time.strftime('%Y%m%d_%H%M')}_{hash(article.headline) % 10000}"
        
        return GeopoliticalEvent(
            event_id=event_id,
            title=article.headline,
            description=article.content[:500],
            country=country,
            region=region,
            severity=risk_level,
            market_impact=self._assess_market_impact(article, risk_level),
            affected_assets=article.affected_instruments,
            probability=min(1.0, severity_score / 10.0),
            start_date=article.published_time,
            end_date=None,
            sources=[article.source]
        )
        
    def _extract_country(self, text: str) -> str:
        """从文本中提取国家名"""
        countries = [
            'china', 'russia', 'ukraine', 'iran', 'north korea', 'syria',
            'israel', 'palestine', 'turkey', 'saudi arabia', 'india',
            'pakistan', 'afghanistan', 'iraq', 'libya', 'venezuela'
        ]
        
        for country in countries:
            if country in text:
                return country.title()
                
        return "Unknown"
        
    def _get_region_from_country(self, country: str) -> str:
        """根据国家确定地区"""
        region_map = {
            'China': 'Asia',
            'Russia': 'Europe/Asia',
            'Ukraine': 'Europe',
            'Iran': 'Middle East',
            'North Korea': 'Asia',
            'Israel': 'Middle East',
            'Turkey': 'Europe/Middle East',
            'Saudi Arabia': 'Middle East',
            'India': 'Asia',
            'Pakistan': 'Asia'
        }
        
        return region_map.get(country, "Global")
        
    def _assess_market_impact(self, article: NewsEventSentiment, risk_level: RiskLevel) -> str:
        """评估市场影响"""
        impact_map = {
            RiskLevel.LOW: "Minimal market impact expected",
            RiskLevel.MEDIUM: "Moderate volatility possible in related sectors",
            RiskLevel.HIGH: "Significant market volatility likely, flight to safety possible",
            RiskLevel.EXTREME: "Major market disruption expected, widespread risk-off sentiment"
        }
        
        return impact_map.get(risk_level, "Impact unclear")
        
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """验证新闻数据质量"""
        try:
            if not data.get('news_sentiment'):
                return False
                
            # 检查是否有足够的新闻
            return len(data['news_sentiment']) > 0
            
        except Exception as e:
            self.logger.error(f"新闻数据验证失败: {e}")
            return False
            
    def get_rate_limit(self) -> Dict[str, int]:
        """获取速率限制"""
        return {
            'requests_per_minute': 30,
            'requests_per_day': 1000,
            'concurrent_requests': 3
        }
        
    async def _perform_health_check(self) -> bool:
        """执行健康检查"""
        try:
            test_data = await self.fetch_data(
                sources=['reuters_business'], 
                hours_back=24, 
                max_articles=5
            )
            return len(test_data.get('news_sentiment', [])) > 0
            
        except Exception as e:
            self.logger.error(f"新闻数据源健康检查失败: {e}")
            return False
            
    def get_supported_symbols(self) -> List[str]:
        """获取支持的新闻源列表"""
        return list(self.rss_feeds.keys()) + list(self.geopolitical_feeds.keys()) 