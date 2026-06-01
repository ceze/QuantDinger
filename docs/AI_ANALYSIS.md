## POST请求参数

{
    "market": "Crypto",
    "symbol": "XLM/USDT",
    "language": "en-US",
    "timeframe": "1D",
    "async_submit": true
}

*支持的语言
language 值	输出语言
zh-CN	简体中文
zh-TW	繁体中文
en-US	英文（默认）
ja-JP	日文
其他值	英文（fallback）


ai雷达
https://dinger-front123.ktx.app/api/global-market/opportunities?_t=1780233414101
{
    "code": 1,
    "msg": "success",
    "data": [
        {
            "symbol": "BNB",
            "name": "BNB",
            "price": 724.33,
            "change_24h": 7.816,
            "change_7d": 0.0,
            "signal": "bullish_momentum",
            "strength": "medium",
            "reason": "24h\u6da8\u5e457.8%\uff0c\u4e0a\u6da8\u52a8\u80fd\u5f3a\u52b2",
            "impact": "bullish",
            "market": "Crypto",
            "timestamp": 1780232692
        }
    ]
}

 # 所以如果想增加雷达扫描的币种，
 # opportunities.py 第 247 行
 CoinGecko Top 20 出现在探测雷达里，修改这里100

# crypto.py里 fetch_crypto_prices_ccxt 
# 币种价格从coingecko拉取，数量per_page 改为300
url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 300,
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "24h,7d",
        }
https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=30&page=1&sparkline=false&price_change_percentage=24h%2C7d




docker buildx build --platform linux/amd64 -t registry-intl.cn-hongkong.aliyuncs.com/madex/quantdinger-linux:latest .
