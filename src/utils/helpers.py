"""通用工具函数"""

import asyncio
import hashlib
import json
import math
import statistics
from datetime import datetime, date, timedelta, timezone
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from functools import wraps
import numpy as np
import pandas as pd
from loguru import logger


# 日期和时间工具
def get_market_timezone():
    """获取美国东部时区"""
    import pytz
    return pytz.timezone('America/New_York')


def to_market_time(dt: datetime) -> datetime:
    """转换为美国东部时间"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(get_market_timezone())


def is_market_hours(dt: Optional[datetime] = None) -> bool:
    """检查是否在市场交易时间内"""
    if dt is None:
        dt = datetime.now()
    
    market_time = to_market_time(dt)
    
    # 检查是否为工作日（周一到周五）
    if market_time.weekday() >= 5:
        return False
    
    # 常规交易时间 9:30 - 16:00 ET
    market_open = market_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = market_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= market_time <= market_close


def is_extended_hours(dt: Optional[datetime] = None) -> bool:
    """检查是否在盘前盘后时间"""
    if dt is None:
        dt = datetime.now()
        
    market_time = to_market_time(dt)
    
    # 检查是否为工作日
    if market_time.weekday() >= 5:
        return False
    
    # 盘前时间 4:00 - 9:30 ET
    premarket_start = market_time.replace(hour=4, minute=0, second=0, microsecond=0)
    market_open = market_time.replace(hour=9, minute=30, second=0, microsecond=0)
    
    # 盘后时间 16:00 - 20:00 ET
    market_close = market_time.replace(hour=16, minute=0, second=0, microsecond=0)
    aftermarket_end = market_time.replace(hour=20, minute=0, second=0, microsecond=0)
    
    return (premarket_start <= market_time < market_open) or (market_close < market_time <= aftermarket_end)


def get_next_market_day(dt: Optional[date] = None) -> date:
    """获取下一个交易日"""
    if dt is None:
        dt = date.today()
    
    next_day = dt + timedelta(days=1)
    
    # 跳过周末
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    
    # TODO: 添加假期检查
    
    return next_day


def get_business_days_between(start: date, end: date) -> int:
    """计算两个日期之间的交易日数量"""
    days = 0
    current = start
    
    while current <= end:
        if current.weekday() < 5:  # 周一到周五
            days += 1
        current += timedelta(days=1)
    
    return days


# 数学和统计工具
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法，避免除零错误"""
    if denominator == 0:
        return default
    return numerator / denominator


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """计算百分比变化"""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def calculate_volatility(prices: List[float], periods: int = 252) -> float:
    """计算年化波动率"""
    if len(prices) < 2:
        return 0.0
    
    returns = []
    for i in range(1, len(prices)):
        returns.append(math.log(prices[i] / prices[i-1]))
    
    if not returns:
        return 0.0
    
    return statistics.stdev(returns) * math.sqrt(periods)


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """计算夏普比率"""
    if not returns:
        return 0.0
    
    excess_returns = [r - risk_free_rate/252 for r in returns]  # 假设日收益率
    
    if statistics.stdev(excess_returns) == 0:
        return 0.0
    
    return statistics.mean(excess_returns) / statistics.stdev(excess_returns) * math.sqrt(252)


def calculate_moving_average(prices: List[float], window: int) -> List[float]:
    """计算移动平均线"""
    if len(prices) < window:
        return []
    
    ma_values = []
    for i in range(window - 1, len(prices)):
        ma_values.append(sum(prices[i-window+1:i+1]) / window)
    
    return ma_values


def moving_average(values: List[float]) -> Optional[float]:
    """计算移动平均值（单个值）"""
    if not values:
        return None
    return sum(values) / len(values)


def calculate_bollinger_bands(prices: List[float], window: int = 20, num_std: float = 2.0) -> Dict[str, List[float]]:
    """计算布林带"""
    if len(prices) < window:
        return {'upper': [], 'middle': [], 'lower': []}
    
    middle = calculate_moving_average(prices, window)
    upper = []
    lower = []
    
    for i in range(window - 1, len(prices)):
        price_slice = prices[i-window+1:i+1]
        std = statistics.stdev(price_slice)
        ma = middle[i-window+1]
        
        upper.append(ma + num_std * std)
        lower.append(ma - num_std * std)
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }


# 数据转换工具
def convert_to_numeric(value: Any, default: float = 0.0) -> float:
    """转换为数值类型"""
    try:
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # 移除逗号和其他格式字符
            cleaned = value.replace(',', '').replace('$', '').replace('%', '')
            return float(cleaned)
        else:
            return default
    except (ValueError, TypeError):
        return default


def format_currency(value: float, currency: str = 'USD') -> str:
    """格式化货币"""
    if currency == 'USD':
        return f"${value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """格式化百分比"""
    return f"{value:.{decimal_places}f}%"


def format_large_number(value: float) -> str:
    """格式化大数字"""
    if abs(value) >= 1e9:
        return f"{value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.1f}K"
    else:
        return f"{value:.1f}"


# 缓存和性能工具
def create_cache_key(*args, **kwargs) -> str:
    """创建缓存键"""
    # 将参数转换为字符串并创建哈希
    cache_data = {
        'args': args,
        'kwargs': kwargs
    }
    cache_str = json.dumps(cache_data, sort_keys=True, default=str)
    return hashlib.md5(cache_str.encode()).hexdigest()


def async_retry(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """异步重试装饰器"""
    def decorator(func: Callable[..., Awaitable]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries} attempts failed. Last error: {e}")
            
            raise last_exception
        return wrapper
    return decorator


async def run_with_timeout(coro: Awaitable, timeout: float):
    """运行协程并设置超时"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"Operation timed out after {timeout} seconds")
        raise


# 数据验证工具
def validate_symbol(symbol: str) -> bool:
    """验证股票代码格式"""
    if not symbol or not isinstance(symbol, str):
        return False
    
    # 移除空格并转换为大写
    symbol = symbol.strip().upper()
    
    # 基本格式检查
    if not symbol.isalpha() or len(symbol) < 1 or len(symbol) > 5:
        return False
    
    return True


def validate_price(price: float) -> bool:
    """验证价格"""
    return isinstance(price, (int, float)) and price > 0


def validate_volume(volume: int) -> bool:
    """验证交易量"""
    return isinstance(volume, int) and volume >= 0


def validate_date_range(start_date: date, end_date: date) -> bool:
    """验证日期范围"""
    return start_date <= end_date and end_date <= date.today()


# 数据清洗工具
def clean_market_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """清洗市场数据"""
    cleaned_data = {}
    
    for key, value in data.items():
        if key in ['price', 'open', 'high', 'low', 'close']:
            cleaned_data[key] = convert_to_numeric(value)
        elif key in ['volume', 'shares']:
            cleaned_data[key] = int(convert_to_numeric(value))
        elif key == 'symbol':
            cleaned_data[key] = str(value).upper().strip()
        elif key == 'timestamp':
            if isinstance(value, str):
                cleaned_data[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
            else:
                cleaned_data[key] = value
        else:
            cleaned_data[key] = value
    
    return cleaned_data


def remove_outliers(data: List[float], method: str = 'iqr', threshold: float = 1.5) -> List[float]:
    """移除异常值"""
    if not data:
        return data
    
    if method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        return [x for x in data if lower_bound <= x <= upper_bound]
    
    elif method == 'zscore':
        mean = statistics.mean(data)
        std = statistics.stdev(data)
        
        return [x for x in data if abs((x - mean) / std) <= threshold]
    
    else:
        return data


# 技术指标计算
def calculate_rsi(prices: List[float], window: int = 14) -> List[float]:
    """计算RSI指标"""
    if len(prices) < window + 1:
        return []
    
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [max(0, change) for change in changes]
    losses = [max(0, -change) for change in changes]
    
    rsi_values = []
    
    # 计算初始平均收益和损失
    avg_gain = sum(gains[:window]) / window
    avg_loss = sum(losses[:window]) / window
    
    if avg_loss == 0:
        rsi_values.append(100)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100 - (100 / (1 + rs)))
    
    # 计算后续RSI值
    for i in range(window, len(gains)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window
        
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))
    
    return rsi_values


def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
    """计算MACD指标"""
    if len(prices) < slow:
        return {'macd': [], 'signal': [], 'histogram': []}
    
    # 计算指数移动平均
    def ema(data: List[float], window: int) -> List[float]:
        alpha = 2.0 / (window + 1)
        ema_values = [data[0]]
        
        for i in range(1, len(data)):
            ema_values.append(alpha * data[i] + (1 - alpha) * ema_values[-1])
        
        return ema_values
    
    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)
    
    # 计算MACD线
    macd_line = [fast_ema[i] - slow_ema[i] for i in range(len(slow_ema))]
    
    # 计算信号线
    signal_line = ema(macd_line, signal)
    
    # 计算MACD柱状图
    histogram = [macd_line[i] - signal_line[i] for i in range(len(signal_line))]
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


# 错误处理工具
class ValidationError(Exception):
    """数据验证错误"""
    pass


class DataSourceError(Exception):
    """数据源错误"""
    pass


class CalculationError(Exception):
    """计算错误"""
    pass


def handle_api_error(response_code: int, response_text: str) -> Exception:
    """处理API错误响应"""
    error_messages = {
        400: "Bad Request - 请求参数错误",
        401: "Unauthorized - API密钥无效",
        403: "Forbidden - 访问被拒绝",
        404: "Not Found - 资源不存在",
        429: "Too Many Requests - 请求过于频繁",
        500: "Internal Server Error - 服务器内部错误",
        502: "Bad Gateway - 网关错误",
        503: "Service Unavailable - 服务不可用"
    }
    
    message = error_messages.get(response_code, f"HTTP Error {response_code}")
    return DataSourceError(f"{message}: {response_text}")


# 配置工具
def deep_merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """深度合并字典"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def get_nested_value(data: Dict, path: str, default: Any = None) -> Any:
    """获取嵌套字典中的值"""
    keys = path.split('.')
    current = data
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def set_nested_value(data: Dict, path: str, value: Any) -> None:
    """设置嵌套字典中的值"""
    keys = path.split('.')
    current = data
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value 