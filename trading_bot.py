"""
Bentley's Trading Bot
---------------------
Strategy: moving-average crossover with an RSI filter.
  BUY  when the fast average crosses above the slow average and RSI isn't overbought.
  SELL when the fast average crosses below the slow average, or stop-loss / take-profit hits.

Modes:
  python trading_bot.py backtest --symbol BTC-USD
  python trading_bot.py backtest --symbol AAPL --period 2y
  python trading_bot.py paper    --symbol BTC-USD      (fake money, runs on a loop)
  python trading_bot.py demo                           (offline, made-up prices)

Setup:  pip install yfinance pandas
Paper trading only: no real money is ever touched.
"""

import argparse
import json
import os
import time
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")   # hide yfinance/pandas deprecation spam

import numpy as np
import pandas as pd

TOP_COINS = ["BTC", "ETH", "SOL", "XRP"]
MEME_COINS = ["DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI"]
CRYPTO = set(TOP_COINS + MEME_COINS + ["ADA", "LTC", "BNB", "AVAX", "DOT", "LINK", "TRUMP"])
PUBLIC_EXCHANGE = "kraken"   # where paper mode gets crypto prices from


def fix_symbols(text, quote="USD"):
    """'btc, pepe, AAPL' -> ['BTC/USD', 'PEPE/USD', 'AAPL'].
    Coins become exchange pairs (prices from the exchange); anything else goes to Yahoo (stocks)."""
    out = []
    for s in text.replace(" ", "").upper().split(","):
        if not s or s in out:
            continue
        if "/" in s or "-" in s:
            out.append(s)
        elif s in CRYPTO or is_listed(f"{s}/{quote}"):
            out.append(f"{s}/{quote}")
        else:
            out.append(s)   # treat as a stock (Yahoo)
    return out


def is_listed(pair):
    try:
        return pair in public_exchange().markets
    except Exception:
        return False


def fmt_price(x):
    """0.0000111 instead of 1.11e-05, so tiny meme-coin prices are readable."""
    return f"{x:,.2f}" if x >= 100 else f"{x:.10f}".rstrip("0").rstrip(".") if x < 1 else f"{x:,.4f}"


def run_coins(coins, one_step):
    """Run one check per coin; drop a coin after 3 failures in a row."""
    for s, p in list(coins):
        try:
            one_step(s, p)
            run_coins.fails[s] = 0
        except SystemExit:
            raise
        except Exception as e:
            run_coins.fails[s] = run_coins.fails.get(s, 0) + 1
            print(f"{s}: error {e}")
            if run_coins.fails[s] >= 3:
                coins.remove((s, p))
                print(f"{s}: removed after 3 errors in a row.")
    if not coins:
        raise SystemExit("No coins left to trade.")


run_coins.fails = {}

# ---------------- settings (tweak these) ----------------
CONFIG = {
    "fast_ma": 10,          # fast moving average length
    "slow_ma": 30,          # slow moving average length
    "rsi_period": 14,
    "rsi_max_buy": 70,      # don't buy if RSI above this (overbought)
    "stop_loss": 0.05,      # sell if down 5% from entry
    "take_profit": 0.15,    # sell if up 15% from entry
    "fee": 0.001,           # 0.1% per trade
    "start_cash": 10_000.0,
    "position_size": 1.0,   # fraction of cash to use per buy
}


# ---------------- data ----------------
_public = None


def public_exchange():
    global _public
    if _public is None:
        import ccxt
        _public = getattr(ccxt, PUBLIC_EXCHANGE)({"enableRateLimit": True})
        _public.load_markets()
    return _public


def get_prices(symbol, period="1y", interval="1d"):
    if "/" in symbol:   # crypto pair: use the exchange's public price data (no keys needed)
        bars = public_exchange().fetch_ohlcv(symbol, timeframe=interval, limit=720)
        df = pd.DataFrame(bars, columns=["t", "o", "h", "l", "close", "v"])
        df.index = pd.to_datetime(df["t"], unit="ms")
        return df[["close"]]
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise SystemExit(f"No data for {symbol}. Check the symbol (e.g. AAPL, BTC-USD).")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna().rename("close").to_frame()


def fake_prices(days=500, seed=7):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0006, 0.025, days)
    prices = 100 * np.exp(np.cumsum(steps))
    idx = pd.date_range(end=datetime.today(), periods=days, freq="D")
    return pd.DataFrame({"close": prices}, index=idx)


# ---------------- indicators + signals ----------------
def add_indicators(df, cfg=CONFIG):
    df = df.copy()
    df["fast"] = df["close"].rolling(cfg["fast_ma"]).mean()
    df["slow"] = df["close"].rolling(cfg["slow_ma"]).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(cfg["rsi_period"]).mean()
    loss = (-delta.clip(upper=0)).rolling(cfg["rsi_period"]).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(100)
    above = df["fast"] > df["slow"]
    df["cross_up"] = above & ~above.shift(1, fill_value=False)
    df["cross_down"] = ~above & above.shift(1, fill_value=False)
    return df


def decide(row, holding, entry_price, cfg=CONFIG):
    """Return 'BUY', 'SELL' or 'HOLD' for one bar."""
    price = row["close"]
    if holding:
        change = price / entry_price - 1
        if change <= -cfg["stop_loss"]:
            return "SELL", "stop-loss"
        if change >= cfg["take_profit"]:
            return "SELL", "take-profit"
        if row["cross_down"]:
            return "SELL", "MA cross down"
    elif row["cross_up"] and row["rsi"] < cfg["rsi_max_buy"]:
        return "BUY", "MA cross up"
    return "HOLD", ""


# ---------------- backtest ----------------
def backtest(df, cfg=CONFIG, verbose=True):
    df = add_indicators(df, cfg).dropna(subset=["slow"])
    cash, units, entry = cfg["start_cash"], 0.0, 0.0
    trades, equity = [], []

    for ts, row in df.iterrows():
        action, why = decide(row, units > 0, entry, cfg)
        price = row["close"]
        if action == "BUY":
            spend = cash * cfg["position_size"]
            units = spend * (1 - cfg["fee"]) / price
            cash -= spend
            entry = price
            trades.append({"time": ts, "side": "BUY", "price": price, "why": why})
        elif action == "SELL":
            cash += units * price * (1 - cfg["fee"])
            trades.append({"time": ts, "side": "SELL", "price": price, "why": why,
                           "pnl_pct": (price / entry - 1) * 100})
            units = 0.0
        equity.append(cash + units * price)

    eq = pd.Series(equity, index=df.index)
    final = eq.iloc[-1]
    sells = [t for t in trades if t["side"] == "SELL"]
    wins = [t for t in sells if t["pnl_pct"] > 0]
    drawdown = (eq / eq.cummax() - 1).min() * 100
    buy_hold = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    result = {
        "final_value": round(final, 2),
        "bot_return_pct": round((final / cfg["start_cash"] - 1) * 100, 2),
        "buy_and_hold_pct": round(buy_hold, 2),
        "trades": len(sells),
        "win_rate_pct": round(len(wins) / len(sells) * 100, 1) if sells else 0,
        "max_drawdown_pct": round(drawdown, 2),
    }
    if verbose:
        print("\n=== Trades ===")
        for t in trades:
            extra = f"  ({t['pnl_pct']:+.1f}%)" if "pnl_pct" in t else ""
            print(f"{t['time']:%Y-%m-%d}  {t['side']:4}  @ {t['price']:,.2f}  {t['why']}{extra}")
        print("\n=== Results ===")
        for k, v in result.items():
            print(f"{k:18}: {v}")
    return result


# ---------------- paper trading (fake money, live prices) ----------------
def find_data_dir():
    """Pick a folder Windows lets us save in: next to the bot first, then your user folders."""
    here = os.path.dirname(os.path.abspath(__file__))
    options = ([os.getenv("BOT_DATA_DIR")] if os.getenv("BOT_DATA_DIR") else []) + [
               os.path.join(here, "bot_data"),
               os.path.join(os.getenv("LOCALAPPDATA") or os.path.expanduser("~"), "BentleyTradingBot"),
               os.path.join(os.path.expanduser("~"), "BentleyTradingBot")]
    for d in options:
        try:
            os.makedirs(d, exist_ok=True)
            test = os.path.join(d, ".write_test")
            with open(test, "w") as f:
                f.write("ok")
            os.remove(test)
            return d
        except OSError:
            continue
    raise SystemExit("Windows won't let the bot save files anywhere. Move the bot folder to C:\\TradingBot and try again.")


_data_dir = None


def state_path(prefix, symbol):
    global _data_dir
    if _data_dir is None:
        _data_dir = find_data_dir()
        print(f"Saving progress in: {_data_dir}")
    return os.path.join(_data_dir, f"{prefix}_{symbol.replace('/', '-')}.json")


def load_state(path, start):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"cash": start, "start": start, "units": 0.0, "entry": 0.0, "log": []}


def save_state(state, path):
    tmp = path + ".tmp"   # write then swap, so the web panel never reads half a file
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def paper_step(symbol, interval, path, start):
    period = "5d" if interval.endswith("m") else "6mo"
    df = add_indicators(get_prices(symbol, period=period, interval=interval))
    row = df.iloc[-1]
    state = load_state(path, start)
    action, why = decide(row, state["units"] > 0, state["entry"])
    price = float(row["close"])

    if action == "BUY":
        spend = state["cash"] * CONFIG["position_size"]
        state["units"] = spend * (1 - CONFIG["fee"]) / price
        state["cash"] -= spend
        state["entry"] = price
    elif action == "SELL":
        state["cash"] += state["units"] * price * (1 - CONFIG["fee"])
        state["units"], state["entry"] = 0.0, 0.0

    state["last_price"] = price
    note_signal(state, row)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {symbol:10} {fmt_price(price)}  {action} {why}"
    if action != "HOLD":
        state["log"].append(line)
        print(line)
    save_state(state, path)


def paper_sell_all(symbol, path):
    if not os.path.exists(path):
        return
    state = load_state(path, 0)
    if state["units"] > 0:
        price = state.get("last_price", state["entry"])
        state["cash"] += state["units"] * price * (1 - CONFIG["fee"])
        state["log"].append(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {symbol:10} {fmt_price(price)}  SELL manual")
        state["units"], state["entry"] = 0.0, 0.0
        save_state(state, path)
        print(f"Sold {symbol} at {fmt_price(price)}.")


def check_pairs(symbols, markets):
    ok = [s for s in symbols if "/" not in s or s in markets]
    for s in symbols:
        if s not in ok:
            print(f"Skipping {s}: not available on this exchange.")
    if not ok:
        raise SystemExit("None of those coins are available.")
    return ok


def paper_loop(symbols, interval, every_min, runner=None):
    if any("/" in s for s in symbols):
        symbols = check_pairs(symbols, public_exchange().markets)
    start_each = CONFIG["start_cash"] / len(symbols)
    coins = [(s, state_path("paper", s)) for s in symbols]
    print(f"Paper trading {len(coins)} coins, ${start_each:,.0f} fake money each: "
          + ", ".join(symbols))

    def step():
        run_coins(coins, lambda s, p: paper_step(s, interval, p, start_each))

    (runner or run_with_menu)(step, every_min, coins, "PAPER (fake money)",
                  lambda: [paper_sell_all(s, p) for s, p in coins])


# ---------------- LIVE trading (REAL money, via ccxt) ----------------
# Setup:  pip install ccxt
# Keys go in environment variables, never in this file:
#   EXCHANGE_API_KEY, EXCHANGE_API_SECRET   (and EXCHANGE_API_PASSWORD if your exchange needs one)
# Make the API key "trade only" with withdrawals DISABLED.
LIVE = {
    "max_trade": 50.0,         # most quote currency (e.g. AUD/USD) spent per buy, PER COIN
    "max_total_loss": 100.0,   # bot stops once closed trades across ALL coins lose this much
    "stop_file": "STOP",       # create a file named STOP next to the bot to halt it
}


# ---------------- eToro connection ----------------
# Keys go in environment variables:
#   ETORO_API_KEY        your eToro Public API key
#   ETORO_DEMO_USER_KEY  user key made for the DEMO environment
#   ETORO_REAL_USER_KEY  user key made for the REAL environment
# Make keys in eToro: Settings > Trading > API Key Management > Create New Key.
class EtoroExchange:
    """Makes eToro look like the other exchanges to the rest of the bot."""
    BASE = "https://public-api.etoro.com"
    TIMEFRAMES = {"1m": "OneMinute", "5m": "FiveMinutes", "15m": "FifteenMinutes",
                  "30m": "ThirtyMinutes", "1h": "OneHour", "4h": "FourHours", "1d": "OneDay"}

    def __init__(self, demo=True):
        self.demo = demo
        self.env = "demo/" if demo else ""
        self.api_key = os.getenv("ETORO_API_KEY")
        self.user_key = os.getenv("ETORO_DEMO_USER_KEY" if demo else "ETORO_REAL_USER_KEY") \
            or os.getenv("ETORO_USER_KEY")
        if not self.api_key or not self.user_key:
            which = "ETORO_DEMO_USER_KEY" if demo else "ETORO_REAL_USER_KEY"
            raise SystemExit(f"Set ETORO_API_KEY and {which} first.")
        self.ids = {}   # "BTC" -> instrumentId
        self.markets = EtoroMarkets(self)
        self.track_file = state_path("etoro", "demo" if demo else "real")
        self.track = {}
        if os.path.exists(self.track_file):
            with open(self.track_file) as f:
                self.track = json.load(f)
        self._req("GET", f"/api/v1/trading/info/{self.env}pnl")   # checks the keys work
        print(f"Connected to eToro {'DEMO' if demo else 'REAL'} account.")

    # --- plumbing ---
    def _req(self, method, path, body=None, params=None):
        import urllib.request, urllib.parse, urllib.error, uuid
        url = self.BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(4):
            req = urllib.request.Request(url, data=data, method=method, headers={
                "x-api-key": self.api_key, "x-user-key": self.user_key,
                "x-request-id": str(uuid.uuid4()), "Content-Type": "application/json",
                "Accept": "application/json",
                # without a normal User-Agent, eToro's firewall blocks Python (error 1010)
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BentleyTradingBot/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    txt = r.read().decode()
                    return json.loads(txt) if txt else {}
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 3:   # too many requests: wait and retry
                    time.sleep(5 * (attempt + 1))
                    continue
                detail = e.read().decode(errors="ignore")[:200]
                if "error code: 10" in detail:
                    raise SystemExit(f"eToro's firewall blocked the connection ({detail.strip()}). "
                                     "Try again in a few minutes, or turn off any VPN/proxy.")
                if e.code in (401, 403):
                    raise SystemExit(f"eToro refused the keys ({e.code}). Check them and that the key "
                                     f"is for the {'DEMO' if self.demo else 'REAL'} environment. {detail}")
                raise RuntimeError(f"eToro error {e.code}: {detail}")
        raise RuntimeError("eToro: too many requests, try a slower speed.")

    def _save_track(self):
        save_state(self.track, self.track_file)

    @staticmethod
    def _g(d, *keys, default=None):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return default

    def instrument_id(self, base):
        if base not in self.ids:
            res = self._req("GET", "/api/v1/market-data/search", params={
                "internalSymbolFull": base,
                "fields": "instrumentId,internalSymbolFull,displayname,instrumentType,isOpen"})
            match = [i for i in res.get("items", [])
                     if str(i.get("internalSymbolFull", "")).upper() == base]
            self.ids[base] = int(match[0]["instrumentId"]) if match else None
        return self.ids[base]

    def _positions(self):
        cp = self._req("GET", f"/api/v1/trading/info/{self.env}pnl").get("clientPortfolio", {})
        out = []
        for p in cp.get("positions", []):
            if not self._g(p, "isBuy", default=True):
                continue
            out.append({"id": int(self._g(p, "positionID", "positionId")),
                        "inst": int(self._g(p, "instrumentID", "instrumentId")),
                        "units": float(self._g(p, "units", default=0)),
                        "amount": float(self._g(p, "amount", default=0)),
                        "pnl": self._g(p, "pnL", "pnl", default=None)
                               if not isinstance(p.get("unrealizedPnL"), dict)
                               else p["unrealizedPnL"].get("pnL")})
        return float(cp.get("credit", 0) or 0), out

    # --- the same calls the bot uses for other exchanges ---
    def load_markets(self):
        return self.markets

    def market(self, symbol):
        return {"limits": {"amount": {"min": 0}}}

    def amount_to_precision(self, symbol, amount):
        return f"{amount:.8f}"

    def fetch_ohlcv(self, symbol, timeframe="1m", limit=200):
        inst = self.instrument_id(symbol.split("/")[0])
        tf = self.TIMEFRAMES.get(timeframe, "OneMinute")
        res = self._req("GET", f"/api/v1/market-data/instruments/{inst}/history/candles/asc/{tf}/{min(limit, 1000)}")
        rows = []
        for group in res.get("candles", []):
            for c in group.get("candles", []):
                ts = pd.Timestamp(c["fromDate"]).value // 1_000_000
                rows.append([ts, c["open"], c["high"], c["low"], c["close"], c.get("volume", 0)])
        if not rows:
            raise RuntimeError(f"no price data for {symbol}")
        return rows

    def fetch_ticker(self, symbol):
        inst = self.instrument_id(symbol.split("/")[0])
        res = self._req("GET", f"/api/v1/market-data/instruments/{inst}/history/candles/desc/OneMinute/1")
        return {"last": float(res["candles"][0]["candles"][0]["close"])}

    def fetch_balance(self):
        credit, positions = self._positions()
        bal = {"USD": {"free": credit}}
        for base, inst in self.ids.items():
            if inst is not None:
                bal[base] = {"free": sum(p["units"] for p in positions if p["inst"] == inst)}
        return bal

    def create_market_buy_order(self, symbol, units):
        base = symbol.split("/")[0]
        inst = self.instrument_id(base)
        price = self.fetch_ticker(symbol)["last"]
        amount = round(units * price, 2)   # eToro buys by dollar amount
        before = {p["id"] for p in self._positions()[1] if p["inst"] == inst}
        self.track[symbol] = {"before": sorted(before), "ids": []}
        self._save_track()
        self._req("POST", f"/api/v2/trading/execution/{self.env}orders", body={
            "action": "open", "transaction": "buy", "instrumentId": inst, "orderType": "mkt",
            "leverage": 1, "amount": amount, "orderCurrency": "usd"})
        new = []
        for _ in range(10):   # orders fill a moment later: wait for the new position
            time.sleep(2)
            new = [p for p in self._positions()[1] if p["inst"] == inst and p["id"] not in before]
            if new:
                break
        self.track[symbol]["ids"] = [p["id"] for p in new]
        self._save_track()
        filled = sum(p["units"] for p in new) or units
        cost = sum(p["amount"] for p in new) or amount
        return {"average": cost / filled if filled else price, "cost": cost, "filled": filled}

    def create_market_sell_order(self, symbol, units):
        """Closes only the positions the bot opened for this coin."""
        inst = self.instrument_id(symbol.split("/")[0])
        t = self.track.get(symbol, {})
        open_pos = [p for p in self._positions()[1] if p["inst"] == inst]
        mine = [p for p in open_pos if p["id"] in t.get("ids", [])] or \
               [p for p in open_pos if t.get("before") is not None and p["id"] not in t["before"]]
        if not mine:
            raise RuntimeError(f"can't find the bot's {symbol} position. Close it in the eToro app.")
        price = self.fetch_ticker(symbol)["last"]
        proceeds = 0.0
        key = "InstrumentID" if self.demo else "InstrumentId"
        for p in mine:
            self._req("POST", f"/api/v1/trading/execution/{self.env}market-close-orders/positions/{p['id']}",
                      body={key: inst, "UnitsToDeduct": None})
            proceeds += p["amount"] + float(p["pnl"]) if p["pnl"] is not None else p["units"] * price
        self.track.pop(symbol, None)
        self._save_track()
        return {"cost": proceeds}


class EtoroMarkets(dict):
    """'BTC/USD' in markets -> looks the coin up on eToro (USD only)."""
    def __init__(self, ex):
        super().__init__()
        self.ex = ex

    def __contains__(self, symbol):
        base, _, quote = symbol.partition("/")
        return quote == "USD" and self.ex.instrument_id(base) is not None


def connect(exchange_id, sandbox):
    if exchange_id == "etoro":
        return EtoroExchange(demo=sandbox)
    import ccxt
    key, secret = os.getenv("EXCHANGE_API_KEY"), os.getenv("EXCHANGE_API_SECRET")
    if not key or not secret:
        raise SystemExit("Set EXCHANGE_API_KEY and EXCHANGE_API_SECRET first.")
    ex = getattr(ccxt, exchange_id)({
        "apiKey": key, "secret": secret,
        "password": os.getenv("EXCHANGE_API_PASSWORD"),
        "enableRateLimit": True,
    })
    if sandbox:
        ex.set_sandbox_mode(True)
    ex.load_markets()
    return ex


def live_prices(ex, symbol, timeframe, limit=200):
    bars = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["t", "o", "h", "l", "close", "v"])
    df.index = pd.to_datetime(df["t"], unit="ms")
    return df[["close"]].iloc[:-1]   # drop the unfinished candle


def load_live(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"entry": 0.0, "spent": 0.0, "received": 0.0, "open_cost": 0.0, "units": 0.0, "log": []}


def realized(s):
    return s["received"] - (s["spent"] - s.get("open_cost", 0.0))


def live_step(ex, symbol, timeframe, path, bal):
    """One check for one coin. Returns the action taken."""
    state = load_live(path)
    base, quote = symbol.split("/")
    df = add_indicators(live_prices(ex, symbol, timeframe))
    row = df.iloc[-1].copy()
    price = float(ex.fetch_ticker(symbol)["last"])
    row["close"] = price   # live price so stop-loss/take-profit react on every check
    base_free = float(bal.get(base, {}).get("free") or 0)
    quote_free = float(bal.get(quote, {}).get("free") or 0)
    min_amt = ex.market(symbol)["limits"]["amount"]["min"] or 0
    # only count coins the bot bought itself, never your own coins
    mine = min(base_free, state.get("units", 0.0))
    holding = state["entry"] > 0 and mine > min_amt

    action, why = decide(row, holding, state["entry"] or price)
    units = mine if holding else 0.0
    if action == "BUY":
        spend = min(LIVE["max_trade"], quote_free * CONFIG["position_size"])
        amount = float(ex.amount_to_precision(symbol, spend / price))
        if amount <= min_amt:
            action, why = "HOLD", f"not enough {quote} to buy"
        else:
            order = ex.create_market_buy_order(symbol, amount)
            state["entry"] = float(order.get("average") or price)
            state["open_cost"] = float(order.get("cost") or spend)
            state["spent"] += state["open_cost"]
            units = float(order.get("filled") or amount)
    elif action == "SELL":
        amount = float(ex.amount_to_precision(symbol, mine))
        order = ex.create_market_sell_order(symbol, amount)
        state["received"] += float(order.get("cost") or amount * price)
        state["entry"], state["open_cost"], units = 0.0, 0.0, 0.0

    state["last_price"], state["units"], state["quote"] = price, units, quote
    note_signal(state, row)
    if action != "HOLD":
        line = (f"{datetime.now():%Y-%m-%d %H:%M:%S}  {symbol:10} {fmt_price(price)}  {action} {why}"
                f"  | closed P/L {realized(state):+.2f} {quote}")
        state["log"].append(line)
        print(line)
    save_state(state, path)
    return action


def live_loop(exchange_id, pairs, timeframe, every_min, sandbox, runner=None, confirm=True):
    ex = connect(exchange_id, sandbox)
    for s in pairs:
        if "/" not in s:
            print(f"Skipping {s}: not a coin this bot can trade live.")
    pairs = check_pairs([s for s in pairs if "/" in s], ex.markets)
    tag = f"live_{exchange_id}" + ("_demo" if sandbox else "")   # separate files per account
    coins = [(s, state_path(tag, s)) for s in pairs]
    if exchange_id == "etoro":
        where = "DEMO account (fake money)" if sandbox else "REAL MONEY"
    else:
        where = "SANDBOX (test funds)" if sandbox else "REAL MONEY"
    print(f"\n*** LIVE TRADING on {exchange_id} — {where} ***")
    print(f"Coins: {', '.join(pairs)}")
    print(f"Max {LIVE['max_trade']} per buy per coin (up to {LIVE['max_trade'] * len(pairs):,.0f} "
          f"spent at once). Stops after {LIVE['max_total_loss']} total loss.")
    if confirm and input("Type YES to start: ").strip() != "YES":
        raise SystemExit("Cancelled.")

    def step():
        if os.path.exists(LIVE["stop_file"]):
            raise SystemExit("STOP file found. Bot halted.")
        loss = -sum(realized(load_live(p)) for _, p in coins)
        if loss >= LIVE["max_total_loss"]:
            raise SystemExit("Loss limit reached. Bot halted. Pick 4 to sell anything still held.")
        bal = [ex.fetch_balance()]

        def one(s, p):
            if live_step(ex, s, timeframe, p, bal[0]) != "HOLD":
                bal[0] = ex.fetch_balance()   # refresh after a trade
        run_coins(coins, one)

    def sell_all():
        bal = ex.fetch_balance()
        for s, p in coins:
            state = load_live(p)
            base = s.split("/")[0]
            mine = min(float(bal.get(base, {}).get("free") or 0), state.get("units", 0.0))
            if state["entry"] <= 0 or mine <= (ex.market(s)["limits"]["amount"]["min"] or 0):
                continue
            try:
                amount = float(ex.amount_to_precision(s, mine))
                order = ex.create_market_sell_order(s, amount)
                state["received"] += float(order.get("cost") or 0)
                state["entry"], state["open_cost"], state["units"] = 0.0, 0.0, 0.0
                state["log"].append(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {s:10} SELL manual {amount}")
                save_state(state, p)
                print(f"Sold {amount} {base}.")
            except Exception as e:
                print(f"{s}: sell failed {e}")

    (runner or run_with_menu)(step, every_min, coins, f"LIVE on {exchange_id} — {where}", sell_all)


# ---------------- control menu (runs while trading) ----------------
def note_signal(state, row):
    """Remember how close the averages are, so the menu can show it."""
    if pd.notna(row.get("fast")) and pd.notna(row.get("slow")) and row["slow"]:
        state["prev_gap"] = state.get("gap")
        state["gap"] = float(row["fast"] / row["slow"] - 1)
        state["rsi"] = float(row["rsi"])


def signal_text(st):
    price, entry = st.get("last_price", 0.0), st.get("entry", 0.0)
    if st.get("units") and entry:
        to_stop = price / (entry * (1 - CONFIG["stop_loss"])) - 1
        to_target = entry * (1 + CONFIG["take_profit"]) / price - 1
        return f"stop {to_stop:.2%} away, target {to_target:+.2%}"
    gap = st.get("gap")
    if gap is None:
        return "warming up"
    if gap >= 0:
        return "above line, waits for next dip+cross"
    prev = st.get("prev_gap")
    trend = "" if prev is None else (", rising" if gap > prev else ", falling")
    hot = " (overbought)" if st.get("rsi", 0) >= CONFIG["rsi_max_buy"] else ""
    return f"{abs(gap):.2%} to buy{trend}{hot}"


def show_status(coins, label, running):
    print(f"\n====== STATUS: {label} — bot {'RUNNING' if running else 'STOPPED'} ======")
    print(f"{'Coin':11}{'Price':>14}  {'Holding':16}{'Signal':38}{'P/L':>10}")
    total, cur, trades = 0.0, "", 0
    for s, p in coins:
        if not os.path.exists(p):
            print(f"{s:11}{'(waiting for first check)':>14}")
            continue
        with open(p) as f:
            st = json.load(f)
        price, units = st.get("last_price", 0.0), st.get("units", 0.0)
        if "cash" in st:   # paper
            pl = st["cash"] + units * price - st["start"]
            cur = "$"
        else:              # live
            pl = realized(st) + (units * price - st.get("open_cost", 0.0) if units else 0.0)
            cur = st.get("quote", "") + " "
        if units:
            pct = (price / st["entry"] - 1) * 100 if st.get("entry") else 0
            hold = f"yes {pct:+.2f}%"
        else:
            hold = "no"
        total += pl
        trades += len(st.get("log", []))
        print(f"{s:11}{fmt_price(price):>14}  {hold:16}{signal_text(st):38}{pl:>+10,.2f}")
    word = "PROFIT" if total >= 0 else "LOSS"
    print(f"\nTOTAL {word}: {'-' if total < 0 else '+'}{cur}{abs(total):,.2f}   ({trades} trades)")


def show_history(coins):
    log = []
    for _, p in coins:
        if os.path.exists(p):
            with open(p) as f:
                log += json.load(f).get("log", [])
    print("\n====== TRADE HISTORY (newest last) ======")
    print("\n".join(sorted(log)[-20:]) if log else "No trades yet.")


def run_with_menu(step, every_min, coins, label, sell_all):
    import threading
    stop = threading.Event()
    lock = threading.Lock()

    def worker():
        while not stop.is_set():
            try:
                with lock:
                    step()
            except SystemExit as e:
                print(f"\n{e}")
                stop.set()
                break
            except Exception as e:
                print("\nError:", e)
            stop.wait(every_min * 60)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(3)
    while True:
        print("\n--- MENU ---  1) Status / profit & loss   2) Trade history"
              "   3) Stop trading   4) Sell everything & stop")
        try:
            choice = input("Pick 1-4 (Enter to refresh): ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "3"
        if choice in ("", "1"):
            show_status(coins, label, not stop.is_set())
        elif choice == "2":
            show_history(coins)
        elif choice in ("3", "4"):
            stop.set()
            with lock:   # wait for any trade in progress to finish
                if choice == "4":
                    try:
                        sell_all()
                    except Exception as e:
                        print("Sell failed:", e)
            show_status(coins, label, False)
            print("\nTrading stopped.")
            return


# ---------------- command line ----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bentley's trading bot")
    p.add_argument("mode", nargs="?", choices=["backtest", "paper", "demo", "live"])
    p.add_argument("--exchange", default="kraken", help="live mode: ccxt exchange id")
    p.add_argument("--quote", default="USD", help="currency to trade coins with: USD, AUD, USDT ...")
    p.add_argument("--sandbox", action="store_true", help="live mode: use exchange test network")
    p.add_argument("--symbol", default="BTC", help="one or more, comma-separated: BTC,ETH,PEPE or AAPL")
    p.add_argument("--period", default="1y")
    p.add_argument("--interval", default="1d", help="1d, 1h, 15m ...")
    p.add_argument("--every", type=float, default=60, help="minutes between checks (0.5 = 30 sec)")
    a = p.parse_args()

    # Double-clicked with no command? Show a menu instead of closing.
    if a.mode is None:
        print("Bentley's Trading Bot\n 1) Demo (offline)\n 2) Backtest\n 3) Paper trade"
              "\n 4) LIVE (real money, Kraken etc.)"
              "\n 5) eToro DEMO account (fake money)\n 6) eToro REAL money")
        choice = input("Pick 1-6: ").strip()
        a.mode = {"1": "demo", "2": "backtest", "3": "paper", "4": "live",
                  "5": "live", "6": "live"}.get(choice, "demo")
        if choice in ("5", "6"):
            a.exchange, a.sandbox = "etoro", choice == "5"
        elif a.mode == "live":
            a.exchange = input("Exchange [kraken]: ").strip() or "kraken"
        if a.mode == "backtest":
            a.symbol = input("Coin or stock [BTC]: ").strip() or "BTC"
        if a.mode in ("paper", "live"):
            print("Coins:\n 1) Just Bitcoin"
                  f"\n 2) Top coins  ({', '.join(TOP_COINS)})"
                  f"\n 3) Meme coins ({', '.join(MEME_COINS)})"
                  "\n 4) Top + meme coins"
                  "\n 5) Type my own (e.g. BTC,PEPE,DOGE)")
            c = input("Pick 1-5 [1]: ").strip()
            a.symbol = {"2": ",".join(TOP_COINS), "3": ",".join(MEME_COINS),
                        "4": ",".join(TOP_COINS + MEME_COINS)}.get(c, "BTC")
            if c == "5":
                a.symbol = input("Coins: ").strip() or "BTC"
            if a.exchange == "etoro":
                a.quote = "USD"   # eToro trades in US dollars only
            else:
                a.quote = (input("Trade with which currency [USD]: ").strip() or "USD").upper()
            # (candles, minutes between checks, stop-loss, take-profit)
            speeds = {
                "1": ("1m", 0.25, 0.005, 0.01),   # Turbo: every 15 sec
                "2": ("1m", 0.5, 0.01, 0.02),     # Fast: every 30 sec
                "3": ("5m", 1, 0.02, 0.04),       # Medium: every minute
                "4": ("1h", 15, 0.05, 0.15),      # Slow: every 15 min
            }
            print("Speed:\n 1) Turbo  (1-min candles, checks every 15 sec)"
                  "\n 2) Fast   (1-min candles, checks every 30 sec)"
                  "\n 3) Medium (5-min candles, checks every minute)"
                  "\n 4) Slow   (1-hour candles, checks every 15 min)")
            sp = speeds.get(input("Pick 1-4 [2]: ").strip(), speeds["2"])
            a.interval, a.every, CONFIG["stop_loss"], CONFIG["take_profit"] = sp
            print(f"Using {a.interval} candles, stop-loss {sp[2]:.1%}, take-profit {sp[3]:.1%}")

    symbols = fix_symbols(a.symbol, a.quote)
    try:
        if a.mode == "demo":
            backtest(fake_prices())
        elif a.mode == "backtest":
            print(f"Backtesting {symbols[0]}")
            backtest(get_prices(symbols[0], a.period, a.interval))
        elif a.mode == "live":
            live_loop(a.exchange, [s for s in symbols if "/" in s], a.interval, a.every, a.sandbox)
        else:
            paper_loop(symbols, a.interval, a.every)
    except (SystemExit, KeyboardInterrupt) as e:
        if str(e):
            print(e)
    except ModuleNotFoundError as e:
        print(f"Missing package: {e.name}. Run run_bot.bat once, or: pip install yfinance pandas ccxt")
    except Exception as e:
        print("Error:", e)
    try:
        input("\nPress Enter to close...")
    except EOFError:
        pass
