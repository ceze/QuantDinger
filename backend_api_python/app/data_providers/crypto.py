"""Crypto price data fetchers with multi-source fallback."""
from __future__ import annotations

import requests
from typing import Any, Dict, List

from app.utils.logger import get_logger
from app.data_providers import safe_float

logger = get_logger(__name__)

TOP_CRYPTO_SYMBOLS = [
    {"yf": "BTC-USD", "symbol": "BTC", "name": "Bitcoin"},
    {"yf": "ETH-USD", "symbol": "ETH", "name": "Ethereum"},
    {"yf": "BNB-USD", "symbol": "BNB", "name": "Binance Coin"},
    {"yf": "SOL-USD", "symbol": "SOL", "name": "Solana"},
    {"yf": "XRP-USD", "symbol": "XRP", "name": "Ripple"},
    {"yf": "ADA-USD", "symbol": "ADA", "name": "Cardano"},
    {"yf": "DOGE-USD", "symbol": "DOGE", "name": "Dogecoin"},
    {"yf": "AVAX-USD", "symbol": "AVAX", "name": "Avalanche"},
    {"yf": "DOT-USD", "symbol": "DOT", "name": "Polkadot"},
    {"yf": "POL-USD", "symbol": "POL", "name": "Polygon"},
    {"yf": "LINK-USD", "symbol": "LINK", "name": "Chainlink"},
    {"yf": "LTC-USD", "symbol": "LTC", "name": "Litecoin"},
]


def _enrich_change_7d_from_coingecko(result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Supplement change_7d for CCXT results via a single CoinGecko request."""
    if not result:
        return result
    try:
        # Build symbol→index map for fast lookup
        sym_map = {}
        for i, coin in enumerate(result):
            s = (coin.get("symbol") or "").upper()
            if s:
                sym_map[s] = i

        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "7d",
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        for coin in resp.json():
            sym = (coin.get("symbol") or "").upper()
            idx = sym_map.get(sym)
            if idx is not None:
                val = safe_float(coin.get("price_change_percentage_7d_in_currency"))
                if val is not None:
                    result[idx]["change_7d"] = val
    except Exception as e:
        logger.debug("CoinGecko change_7d enrichment failed: %s", e)
    return result


def fetch_crypto_prices_ccxt() -> List[Dict[str, Any]]:
    """Fetch crypto prices using CCXT (system's existing data source)."""
    try:
        from app.data_sources.crypto import CryptoDataSource

        crypto_source = CryptoDataSource()

        symbols = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT", "TRX/USDT", "DOGE/USDT", "ADA/USDT",
            "LINK/USDT", "AVAX/USDT", "XLM/USDT", "TON/USDT", "SHIB/USDT", "HBAR/USDT", "SUI/USDT", "BCH/USDT", "DOT/USDT",
            "LTC/USDT", "PEPE/USDT", "UNI/USDT", "APT/USDT", "NEAR/USDT", "ICP/USDT", "ETC/USDT", "ONDO/USDT", "AAVE/USDT", "POL/USDT",
            "MNT/USDT", "VET/USDT", "RENDER/USDT", "ATOM/USDT", "FIL/USDT", "ALGO/USDT", "ARB/USDT", "OP/USDT", "SEI/USDT", "INJ/USDT",
            "FET/USDT", "TAO/USDT", "TIA/USDT", "IMX/USDT", "JUP/USDT", "WLD/USDT", "BONK/USDT", "GRT/USDT", "MKR/USDT", "QNT/USDT",
            "FLOW/USDT", "SAND/USDT", "MANA/USDT", "THETA/USDT", "KAS/USDT", "PYTH/USDT", "RUNE/USDT", "FLR/USDT", "JASMY/USDT", "BRETT/USDT",
            "STRK/USDT", "ENA/USDT", "LDO/USDT", "EGLD/USDT", "KCS/USDT", "AXS/USDT", "BEAM/USDT", "CFX/USDT", "CHZ/USDT", "GALA/USDT",
            "DYDX/USDT", "PENDLE/USDT", "AKT/USDT", "CRV/USDT", "SNX/USDT", "COMP/USDT", "ZEC/USDT", "ROSE/USDT", "ZRO/USDT", "1INCH/USDT",
            "MINA/USDT", "CKB/USDT", "KAVA/USDT", "IOTA/USDT", "WIF/USDT", "NOT/USDT", "ORDI/USDT", "APE/USDT", "GMX/USDT", "BLUR/USDT",
            "SAFE/USDT", "ARKM/USDT", "ALT/USDT", "ZK/USDT", "SUPER/USDT", "CELO/USDT", "DASH/USDT", "BAT/USDT", "ANKR/USDT", "HOT/USDT"
        ]

        result = []
        for symbol in symbols:
            try:
                ticker = crypto_source.get_ticker(symbol)
                if ticker:
                    base = symbol.split("/")[0]
                    result.append({
                        "symbol": base,
                        "name": base,
                        "price": safe_float(ticker.get("last") or ticker.get("close")),
                        "change_24h": safe_float(ticker.get("percentage", 0)),
                        "change_7d": 0,  # enriched by CoinGecko below
                        "market_cap": 0,
                        "volume_24h": safe_float(ticker.get("quoteVolume", 0)),
                        "image": "",
                        "category": "crypto",
                    })
            except Exception as e:
                logger.debug("Failed to fetch %s: %s", symbol, e)
                continue

        result = _enrich_change_7d_from_coingecko(result)
        return result
    except Exception as e:
        logger.error("Failed to fetch crypto prices via CCXT: %s", e)
        return []


def fetch_crypto_prices_yfinance() -> List[Dict[str, Any]]:
    """Fetch crypto prices using yfinance as fallback."""
    try:
        import yfinance as yf

        symbols = TOP_CRYPTO_SYMBOLS

        yf_symbols = [s["yf"] for s in symbols]
        tickers = yf.Tickers(" ".join(yf_symbols))

        result = []
        for crypto in symbols:
            try:
                ticker = tickers.tickers.get(crypto["yf"])
                if ticker:
                    hist = ticker.history(period="2d")
                    if len(hist) >= 2:
                        prev = hist["Close"].iloc[-2]
                        curr = hist["Close"].iloc[-1]
                        change = ((curr - prev) / prev) * 100
                        result.append({
                            "symbol": crypto["symbol"],
                            "name": crypto["name"],
                            "price": round(curr, 2),
                            "change_24h": round(change, 2),
                            "change_7d": 0,
                            "market_cap": 0,
                            "volume_24h": 0,
                            "image": "",
                            "category": "crypto",
                        })
                    elif len(hist) == 1:
                        result.append({
                            "symbol": crypto["symbol"],
                            "name": crypto["name"],
                            "price": round(hist["Close"].iloc[-1], 2),
                            "change_24h": 0,
                            "change_7d": 0,
                            "market_cap": 0,
                            "volume_24h": 0,
                            "image": "",
                            "category": "crypto",
                        })
            except Exception as e:
                logger.debug("Failed to fetch %s: %s", crypto["yf"], e)

        return result
    except Exception as e:
        logger.error("Failed to fetch crypto via yfinance: %s", e)
        return []


def fetch_crypto_prices(*, fast: bool = False) -> List[Dict[str, Any]]:
    """Fetch top crypto prices — try CoinGecko → CCXT fallback."""

    # if not fast:
    #     result = fetch_crypto_prices_ccxt()
    #     if result and len(result) >= 50:
    #         logger.info("Fetched %d crypto prices via CCXT (fallback)", len(result))
    #         return result

    # CoinGecko 首选：返回完整的 change_24h + change_7d + market_cap
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "24h,7d",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        result = []
        for coin in data:
            result.append({
                "symbol": coin.get("symbol", "").upper(),
                "name": coin.get("name", ""),
                "price": safe_float(coin.get("current_price")),
                "change_24h": safe_float(coin.get("price_change_percentage_24h")),
                "change_7d": safe_float(coin.get("price_change_percentage_7d_in_currency")),
                "market_cap": safe_float(coin.get("market_cap")),
                "volume_24h": safe_float(coin.get("total_volume")),
                "image": coin.get("image", ""),
                "category": "crypto",
            })
        logger.info("Fetched %d crypto prices via CoinGecko", len(result))
        return result
    except Exception as e:
        logger.warning("CoinGecko failed: %s, falling back to CCXT", e)

    # CCXT 回退：缺少 change_7d，尝试 CoinGecko 补充
    result = fetch_crypto_prices_ccxt()
    if result and len(result) >= 5:
        logger.info("Fetched %d crypto prices via CCXT (fallback)", len(result))
        return result

    logger.warning("All crypto data sources failed, returning placeholder data")
    return [
        {"symbol": s["symbol"], "name": s["name"], "price": 0, "change_24h": 0, "change_7d": 0,
         "market_cap": 0, "volume_24h": 0, "image": "", "category": "crypto"}
        for s in TOP_CRYPTO_SYMBOLS
    ]


# ---------- Heatmap-specific fetchers ----------

def fetch_crypto_heatmap_coingecko() -> List[Dict[str, Any]]:
    """Fetch crypto heatmap from CoinGecko with retry."""
    for attempt in range(2):
        try:
            resp = requests.get("https://api.coingecko.com/api/v3/coins/markets", params={
                "vs_currency": "usd", "order": "market_cap_desc",
                "per_page": 30, "page": 1, "sparkline": "false",
                "price_change_percentage": "24h",
            }, timeout=15)
            resp.raise_for_status()
            data = resp.json() or []
            result = []
            for coin in data:
                result.append({
                    "symbol": (coin.get("symbol") or "").upper(),
                    "name": coin.get("name", ""),
                    "price": safe_float(coin.get("current_price")),
                    "change_24h": safe_float(coin.get("price_change_percentage_24h")),
                    "market_cap": safe_float(coin.get("market_cap")),
                    "volume_24h": safe_float(coin.get("total_volume")),
                    "image": coin.get("image", ""),
                    "category": "crypto",
                })
            if result:
                logger.info("Fetched crypto heatmap via CoinGecko: %d items", len(result))
                return result
        except Exception as e:
            logger.debug("CoinGecko attempt %d failed: %s", attempt + 1, e)
            if attempt == 0:
                import time as _t
                _t.sleep(1)
    return []


def fetch_crypto_heatmap_coincap() -> List[Dict[str, Any]]:
    """Fetch crypto heatmap from CoinCap API (free, no key needed)."""
    try:
        resp = requests.get("https://api.coincap.io/v2/assets", params={"limit": 30}, timeout=15)
        resp.raise_for_status()
        data = (resp.json() or {}).get("data") or []
        result = []
        for coin in data:
            result.append({
                "symbol": (coin.get("symbol") or "").upper(),
                "name": coin.get("name", ""),
                "price": safe_float(coin.get("priceUsd")),
                "change_24h": safe_float(coin.get("changePercent24Hr")),
                "market_cap": safe_float(coin.get("marketCapUsd")),
                "volume_24h": safe_float(coin.get("volumeUsd24Hr")),
                "image": "",
                "category": "crypto",
            })
        if result:
            logger.info("Fetched crypto heatmap via CoinCap: %d items", len(result))
        return result
    except Exception as e:
        logger.debug("CoinCap failed: %s", e)
        return []
