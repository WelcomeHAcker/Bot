"""
Bentley's Trading Bot — web control panel
-----------------------------------------
On your PC:   double-click run_web.bat, then open http://localhost:8000
In the cloud: runs 24/7 on a server (see the setup steps), open its web address from anywhere.

Settings (environment variables):
  WEB_PASSWORD   password for the panel (required in the cloud)
  BOT_DATA_DIR   where progress is saved (in the cloud: /data)
  PORT           set automatically by the cloud host
  AUTO_RESUME    "0" to stop the bot restarting itself after the server restarts
Exchange keys are the same as before: EXCHANGE_API_KEY / EXCHANGE_API_SECRET (Kraken),
ETORO_API_KEY / ETORO_DEMO_USER_KEY / ETORO_REAL_USER_KEY (eToro).
"""

import collections
import io
import json
import os
import secrets
import sys
import threading
import time
from datetime import datetime

from flask import Flask, Response, jsonify, request

import trading_bot as tb

# ---------------- keep the bot's printed lines for the web log ----------------
LOG = collections.deque(maxlen=300)


class Tee(io.TextIOBase):
    def __init__(self, orig):
        self.orig, self.buf = orig, ""

    def write(self, text):
        try:
            self.orig.write(text)
        except Exception:
            pass
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                LOG.append(f"{datetime.now():%H:%M:%S}  {line.strip()}")
        return len(text)

    def flush(self):
        try:
            self.orig.flush()
        except Exception:
            pass


sys.stdout = Tee(sys.stdout)

# ---------------- choices shown on the page ----------------
SPEEDS = {   # candles, minutes between checks, stop-loss, take-profit
    "turbo": ("1m", 0.25, 0.005, 0.01),
    "fast": ("1m", 0.5, 0.01, 0.02),
    "medium": ("5m", 1, 0.02, 0.04),
    "slow": ("1h", 15, 0.05, 0.15),
}
PRESETS = {"btc": ["BTC"], "top": tb.TOP_COINS, "meme": tb.MEME_COINS,
           "all": tb.TOP_COINS + tb.MEME_COINS}
MODES = {   # exchange, demo/sandbox, real money?
    "paper": (None, False, False),
    "etoro_demo": ("etoro", True, False),
    "kraken": ("kraken", False, True),
    "etoro_real": ("etoro", False, True),
}


def data_dir():
    tb.state_path("x", "y")   # makes sure the save folder is chosen
    return tb._data_dir


def config_file():
    return os.path.join(data_dir(), "web_config.json")


# ---------------- the bot, run in the background ----------------
class Bot:
    def __init__(self):
        self.lock = threading.Lock()        # one trading check at a time
        self.control = threading.Lock()     # one start/stop at a time
        self.stop_ev = threading.Event()
        self.thread = None
        self.coins, self.label, self.step, self.sell_fn = [], "", None, None
        self.cfg, self.error, self.started, self.last_check = None, None, None, None

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, cfg):
        with self.control:
            if self.running():
                raise ValueError("The bot is already running. Stop it first.")
            mode = cfg.get("mode")
            if mode not in MODES or cfg.get("speed") not in SPEEDS:
                raise ValueError("Pick an account and a speed.")
            exchange, sandbox, real = MODES[mode]
            interval, every, sl, tp = SPEEDS[cfg["speed"]]
            tb.CONFIG["stop_loss"], tb.CONFIG["take_profit"] = sl, tp
            if real:
                max_trade = float(cfg.get("max_trade") or 50)
                max_loss = float(cfg.get("max_loss") or 100)
                if not (1 <= max_trade <= 10_000 and 1 <= max_loss <= 100_000):
                    raise ValueError("Max per buy and loss limit must be sensible numbers.")
                tb.LIVE["max_trade"], tb.LIVE["max_total_loss"] = max_trade, max_loss
            else:
                tb.LIVE["max_trade"], tb.LIVE["max_total_loss"] = 50.0, 100.0
            preset = cfg.get("coins", "btc")
            text = cfg.get("custom", "") if preset == "custom" else ",".join(PRESETS.get(preset, ["BTC"]))
            if not text.strip():
                raise ValueError("Type at least one coin.")
            quote = "USD" if exchange == "etoro" else (cfg.get("quote") or "USD").upper()
            symbols = tb.fix_symbols(text, quote)

            got = {}

            def runner(step, every_min, coins, label, sell_all):
                got.update(step=step, coins=coins, label=label, sell_all=sell_all)

            tb.run_coins.fails = {}
            try:
                if mode == "paper":
                    tb.paper_loop(symbols, interval, every, runner=runner)
                else:
                    tb.live_loop(exchange, symbols, interval, every, sandbox, runner=runner, confirm=False)
            except SystemExit as e:
                raise ValueError(str(e))

            self.step, self.coins, self.label, self.sell_fn = got["step"], got["coins"], got["label"], got["sell_all"]
            self.cfg, self.error, self.started = dict(cfg), None, datetime.now()
            self.stop_ev = threading.Event()
            self.thread = threading.Thread(target=self._worker, args=(every,), daemon=True)
            self.thread.start()
            save_config(self.cfg, running=True)
            print(f"Started: {self.label}, {len(self.coins)} coins, speed {cfg['speed']}.")

    def _worker(self, every):
        while not self.stop_ev.is_set():
            try:
                with self.lock:
                    if self.stop_ev.is_set():
                        break
                    self.step()
                    self.last_check = datetime.now()
            except SystemExit as e:
                self.error = str(e)
                print(e)
                save_config(self.cfg, running=False)
                break
            except Exception as e:
                print("Error:", e)
            self.stop_ev.wait(every * 60)

    def stop(self, sell=False):
        with self.control:
            self.stop_ev.set()
            with self.lock:   # wait for any trade in progress to finish
                if sell and self.sell_fn:
                    try:
                        self.sell_fn()
                    except Exception as e:
                        print("Sell failed:", e)
            if self.thread:
                self.thread.join(timeout=10)
            if self.cfg:
                save_config(self.cfg, running=False)
            print("Trading stopped." + (" Sold everything first." if sell else ""))

    def status(self):
        rows, total, trades, cur, history = [], 0.0, 0, "$", []
        for s, p in list(self.coins):
            st = None
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        st = json.load(f)
                except (OSError, ValueError):
                    st = None
            if st is None:
                rows.append({"coin": s, "price": "", "holding": "", "signal": "waiting for first check", "pl": 0})
                continue
            price, units = st.get("last_price", 0.0), st.get("units", 0.0)
            if "cash" in st:
                pl = st["cash"] + units * price - st["start"]
            else:
                pl = tb.realized(st) + (units * price - st.get("open_cost", 0.0) if units else 0.0)
                cur = st.get("quote", "USD") + " "
            if units and st.get("entry"):
                hold = f"yes {(price / st['entry'] - 1) * 100:+.2f}%"
            else:
                hold = "no"
            total += pl
            trades += len(st.get("log", []))
            history += st.get("log", [])
            rows.append({"coin": s, "price": tb.fmt_price(price) if price else "", "holding": hold,
                         "signal": tb.signal_text(st), "pl": round(pl, 2)})
        return {
            "running": self.running(), "label": self.label, "error": self.error,
            "config": self.cfg, "rows": rows, "total": round(total, 2), "currency": cur.strip(),
            "trades": trades, "history": sorted(history)[-40:][::-1],
            "started": self.started.strftime("%d %b %H:%M") if self.started else None,
            "last_check": self.last_check.strftime("%H:%M:%S") if self.last_check else None,
            "log": list(LOG)[-80:][::-1],
            "keys": {"kraken": bool(os.getenv("EXCHANGE_API_KEY") and os.getenv("EXCHANGE_API_SECRET")),
                     "etoro_demo": bool(os.getenv("ETORO_API_KEY") and (os.getenv("ETORO_DEMO_USER_KEY") or os.getenv("ETORO_USER_KEY"))),
                     "etoro_real": bool(os.getenv("ETORO_API_KEY") and (os.getenv("ETORO_REAL_USER_KEY") or os.getenv("ETORO_USER_KEY")))},
        }


def save_config(cfg, running):
    try:
        tb.save_state({"config": cfg, "running": running}, config_file())
    except Exception as e:
        print("Couldn't save settings:", e)


BOT = Bot()

# ---------------- web server ----------------
app = Flask(__name__)
PASSWORD = os.getenv("WEB_PASSWORD", "")
FAILS = {}   # ip -> [count, first_fail_time]


@app.before_request
def check_password():
    if not PASSWORD:
        return None
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()
    n, first = FAILS.get(ip, (0, time.time()))
    if time.time() - first > 600:
        n, first = 0, time.time()
    if n >= 10:
        return Response("Too many wrong passwords. Wait 10 minutes.", 429)
    auth = request.authorization
    if auth and auth.password and secrets.compare_digest(auth.password.encode(), PASSWORD.encode()):
        FAILS.pop(ip, None)
        return None
    if auth:
        FAILS[ip] = (n + 1, first)
    return Response("Password needed.", 401, {"WWW-Authenticate": 'Basic realm="Trading Bot"'})


def json_only():
    # blocks other websites from pressing the buttons for you
    if not request.is_json:
        return jsonify(error="Bad request."), 400
    return None


@app.get("/")
def page():
    return Response(PAGE, mimetype="text/html")


@app.get("/api/status")
def api_status():
    return jsonify(BOT.status())


@app.post("/api/start")
def api_start():
    bad = json_only()
    if bad:
        return bad
    cfg = request.get_json(silent=True) or {}
    if MODES.get(cfg.get("mode"), (0, 0, False))[2] and cfg.get("confirm") != "YES":
        return jsonify(error="Type YES to confirm real-money trading."), 400
    try:
        BOT.start(cfg)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error=f"Couldn't start: {e}"), 500
    return jsonify(ok=True)


@app.post("/api/stop")
def api_stop():
    bad = json_only()
    if bad:
        return bad
    BOT.stop(sell=bool((request.get_json(silent=True) or {}).get("sell")))
    return jsonify(ok=True)


@app.post("/api/reset_paper")
def api_reset_paper():
    bad = json_only()
    if bad:
        return bad
    if BOT.running():
        return jsonify(error="Stop the bot first."), 400
    removed = 0
    for name in os.listdir(data_dir()):
        if name.startswith("paper_") and name.endswith(".json"):
            os.remove(os.path.join(data_dir(), name))
            removed += 1
    if BOT.cfg and BOT.cfg.get("mode") == "paper":
        BOT.coins = []
    print(f"Paper money reset ({removed} coins cleared).")
    return jsonify(ok=True)


def auto_resume():
    if os.getenv("AUTO_RESUME", "1") == "0":
        return
    try:
        with open(config_file()) as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return
    if saved.get("running") and saved.get("config"):
        print("Server restarted: resuming the bot with the last settings.")
        try:
            BOT.start(saved["config"])
        except Exception as e:
            print("Couldn't resume:", e)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trading Bot</title>
<style>
:root{--bg:#0b0f14;--card:#121821;--line:#1f2a36;--text:#e6edf3;--muted:#8b98a7;
--green:#2fd18b;--red:#ff5c6c;--amber:#f5b642;--blue:#4c9dff;--input:#0e141b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:16px}
header{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:16px}
h1{font-size:20px;margin:0;letter-spacing:.2px}
.pill{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;background:var(--card);border:1px solid var(--line);font-weight:600;font-size:13px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--muted)}
.on .dot{background:var(--green);box-shadow:0 0 0 4px rgba(47,209,139,.18)}
.grid{display:grid;grid-template-columns:340px minmax(0,1fr);gap:16px}
.grid>*,.stack>*{min-width:0}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}
.card h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 12px}
label{display:block;font-size:13px;color:var(--muted);margin:10px 0 4px}
select,input{width:100%;padding:10px;border-radius:9px;border:1px solid var(--line);background:var(--input);color:var(--text);font:inherit}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.row2.one{grid-template-columns:1fr}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:16px}
button{padding:11px;border-radius:9px;border:1px solid var(--line);background:#1a2330;color:var(--text);font:600 14px system-ui,sans-serif;cursor:pointer}
button:disabled{opacity:.4;cursor:not-allowed}
.go{background:var(--green);color:#04130c;border-color:var(--green);grid-column:1/-1}
.sell{border-color:var(--red);color:var(--red)}
.note{font-size:12px;color:var(--muted);margin-top:10px}
.warn{background:rgba(255,92,108,.1);border:1px solid rgba(255,92,108,.4);color:#ffb3bb;padding:10px;border-radius:9px;font-size:13px;margin-top:12px}
.err{display:none}
.big{display:flex;gap:24px;flex-wrap:wrap;align-items:flex-end;margin-bottom:14px}
.big .v{font-size:34px;font-weight:700;font-variant-numeric:tabular-nums}
.big .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.up{color:var(--green)}.down{color:var(--red)}
.tbl{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}
th,td{text-align:left;padding:9px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:500;font-size:12px}
td.sig{white-space:normal;min-width:150px;color:#c3cdd8}
td.r,th.r{text-align:right}
.hold{color:var(--amber);font-weight:600}
.list{font:12.5px/1.6 ui-monospace,Consolas,monospace;max-height:260px;overflow:auto;white-space:pre-wrap;color:#c3cdd8;margin:0}
.stack{display:grid;gap:16px}
.keys{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.key{font-size:12px;padding:3px 8px;border-radius:6px;background:var(--input);border:1px solid var(--line);color:var(--muted)}
.key.ok{color:var(--green);border-color:rgba(47,209,139,.4)}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Bentley's Trading Bot</h1>
    <span id="pill" class="pill"><span class="dot"></span><span id="pillText">Loading…</span></span>
  </header>
  <div class="grid">
    <div class="card">
      <h2>Controls</h2>
      <label for="mode">Account</label>
      <select id="mode">
        <option value="paper">Paper — fake money</option>
        <option value="etoro_demo">eToro DEMO — fake money</option>
        <option value="kraken">Kraken — REAL money</option>
        <option value="etoro_real">eToro — REAL money</option>
      </select>
      <label for="coins">Coins</label>
      <select id="coins">
        <option value="btc">Just Bitcoin</option>
        <option value="top">Top coins (BTC, ETH, SOL, XRP)</option>
        <option value="meme">Meme coins (DOGE, SHIB, PEPE, BONK, WIF, FLOKI)</option>
        <option value="all">Top + meme coins</option>
        <option value="custom">Type my own…</option>
      </select>
      <input id="custom" placeholder="e.g. BTC, PEPE, DOGE" style="display:none;margin-top:8px">
      <div class="row2">
        <div><label for="speed">Speed</label>
          <select id="speed">
            <option value="turbo">Turbo (15 sec)</option>
            <option value="fast" selected>Fast (30 sec)</option>
            <option value="medium">Medium (1 min)</option>
            <option value="slow">Slow (15 min)</option>
          </select></div>
        <div id="quoteBox"><label for="quote">Currency</label>
          <select id="quote"><option>USD</option><option>AUD</option><option>USDT</option></select></div>
      </div>
      <div id="limits" class="row2" style="display:none">
        <div><label for="maxTrade">Max per buy, per coin</label><input id="maxTrade" type="number" min="1" value="50"></div>
        <div><label for="maxLoss">Stop after total loss</label><input id="maxLoss" type="number" min="1" value="100"></div>
      </div>
      <div id="realWarn" class="warn" style="display:none">Real money. Trades happen on their own while the bot runs, and you can lose what you put in.</div>
      <div class="btns">
        <button id="startBtn" class="go">Start trading</button>
        <button id="stopBtn">Stop</button>
        <button id="sellBtn" class="sell">Sell all &amp; stop</button>
      </div>
      <button id="resetBtn" style="width:100%;margin-top:8px">Reset paper money to $10,000</button>
      <div id="err" class="warn err"></div>
      <div class="note">API keys found:</div>
      <div class="keys" id="keys"></div>
    </div>
    <div class="stack">
      <div class="card">
        <div class="big">
          <div><div class="k" id="totalK">Total P/L</div><div class="v" id="total">—</div></div>
          <div><div class="k">Trades</div><div class="v" id="trades" style="font-size:24px">0</div></div>
          <div><div class="k">Last check</div><div class="v" id="last" style="font-size:24px">—</div></div>
        </div>
        <div class="tbl"><table>
          <thead><tr><th>Coin</th><th class="r">Price</th><th>Holding</th><th>Signal</th><th class="r">P/L</th></tr></thead>
          <tbody id="rows"><tr><td colspan="5" style="color:var(--muted)">Not started yet. Pick your settings and press Start.</td></tr></tbody>
        </table></div>
      </div>
      <div class="card"><h2>Trade history</h2><pre class="list" id="history">No trades yet.</pre></div>
      <div class="card"><h2>Bot log</h2><pre class="list" id="log"></pre></div>
    </div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
const REAL = {kraken: 1, etoro_real: 1};
let running = false, filled = false;

function syncForm() {
  const m = $('mode').value;
  $('custom').style.display = $('coins').value === 'custom' ? 'block' : 'none';
  const noQuote = m.startsWith('etoro') || m === 'paper';
  $('quoteBox').style.display = noQuote ? 'none' : 'block';
  $('quoteBox').parentElement.classList.toggle('one', noQuote);
  $('limits').style.display = REAL[m] ? 'grid' : 'none';
  $('realWarn').style.display = REAL[m] ? 'block' : 'none';
}
['mode', 'coins'].forEach(id => $(id).addEventListener('change', syncForm));

function showErr(msg) { const e = $('err'); e.textContent = msg || ''; e.style.display = msg ? 'block' : 'none'; }

async function post(url, body) {
  const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || ('Error ' + r.status));
  return j;
}

$('startBtn').onclick = async () => {
  showErr('');
  const cfg = {mode: $('mode').value, coins: $('coins').value, custom: $('custom').value,
               speed: $('speed').value, quote: $('quote').value,
               max_trade: $('maxTrade').value, max_loss: $('maxLoss').value};
  if (REAL[cfg.mode]) {
    const ok = prompt('REAL MONEY. Up to ' + cfg.max_trade + ' per buy per coin, stops after ' + cfg.max_loss + ' total loss.\nType YES to start:');
    if (ok !== 'YES') return;
    cfg.confirm = 'YES';
  }
  $('startBtn').disabled = true; $('startBtn').textContent = 'Starting…';
  try { await post('/api/start', cfg); } catch (e) { showErr(e.message); }
  $('startBtn').textContent = 'Start trading';
  refresh();
};
$('stopBtn').onclick = async () => { try { await post('/api/stop', {sell: false}); } catch (e) { showErr(e.message); } refresh(); };
$('sellBtn').onclick = async () => {
  if (!confirm('Sell everything the bot bought, then stop?')) return;
  try { await post('/api/stop', {sell: true}); } catch (e) { showErr(e.message); } refresh();
};
$('resetBtn').onclick = async () => {
  if (!confirm('Reset all paper (fake) money back to $10,000?')) return;
  try { await post('/api/reset_paper'); } catch (e) { showErr(e.message); } refresh();
};

const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const money = (v, c) => (v < 0 ? '-' : '+') + (c && c !== '$' ? c + ' ' : '$') + Math.abs(v).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});

async function refresh() {
  let s;
  try { const r = await fetch('/api/status'); if (!r.ok) return; s = await r.json(); } catch (e) { return; }
  running = s.running;
  $('pill').className = 'pill' + (running ? ' on' : '');
  $('pillText').textContent = running ? 'RUNNING · ' + s.label : (s.label ? 'STOPPED · ' + s.label : 'STOPPED');
  ['mode', 'coins', 'custom', 'speed', 'quote', 'maxTrade', 'maxLoss', 'startBtn', 'resetBtn'].forEach(id => $(id).disabled = running);
  $('stopBtn').disabled = $('sellBtn').disabled = !running;
  if (s.config && !filled) {
    filled = true;
    for (const [id, k] of [['mode', 'mode'], ['coins', 'coins'], ['custom', 'custom'], ['speed', 'speed'], ['quote', 'quote'], ['maxTrade', 'max_trade'], ['maxLoss', 'max_loss']])
      if (s.config[k] !== undefined && s.config[k] !== '') $(id).value = s.config[k];
    syncForm();
  }
  if (s.error) showErr(s.error);
  $('total').textContent = s.rows.length ? money(s.total, s.currency) : '—';
  $('total').className = 'v ' + (s.total > 0 ? 'up' : s.total < 0 ? 'down' : '');
  $('totalK').textContent = s.total < 0 ? 'Total loss' : 'Total profit';
  $('trades').textContent = s.trades;
  $('last').textContent = s.last_check || '—';
  if (s.rows.length) {
    $('rows').innerHTML = s.rows.map(r => '<tr><td><b>' + esc(r.coin) + '</b></td><td class="r">' + esc(r.price) +
      '</td><td class="' + (r.holding.startsWith('yes') ? 'hold' : '') + '">' + esc(r.holding) + '</td><td class="sig">' + esc(r.signal) +
      '</td><td class="r ' + (r.pl > 0 ? 'up' : r.pl < 0 ? 'down' : '') + '">' + (r.pl >= 0 ? '+' : '') + r.pl.toFixed(2) + '</td></tr>').join('');
  }
  $('history').textContent = s.history.length ? s.history.join('\n') : 'No trades yet.';
  $('log').textContent = s.log.join('\n');
  $('keys').innerHTML = [['kraken', 'Kraken'], ['etoro_demo', 'eToro demo'], ['etoro_real', 'eToro real']]
    .map(([k, n]) => '<span class="key ' + (s.keys[k] ? 'ok' : '') + '">' + (s.keys[k] ? '✓ ' : '✗ ') + n + '</span>').join('');
}
syncForm();
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    cloud = bool(os.getenv("PORT"))
    port = int(os.getenv("PORT", "8000"))
    if cloud and not PASSWORD:
        raise SystemExit("Set WEB_PASSWORD before running in the cloud, or anyone could control your bot.")
    host = "0.0.0.0" if (cloud or PASSWORD) else "127.0.0.1"   # no password = only this PC can open it
    print(f"Saving progress in: {data_dir()}")
    threading.Thread(target=auto_resume, daemon=True).start()
    where = f"http://localhost:{port}" if not cloud else f"port {port}"
    print(f"Web panel running: open {where}" + ("" if PASSWORD else "  (no password, this PC only)"))
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        app.run(host=host, port=port, threaded=True)
