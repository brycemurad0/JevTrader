"""The dashboard's single HTML page: vanilla JS, no build step, inline CSS, dark theme. Served
as-is by `GET /` in `jevtrader.dashboard.app`."""

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JevTrader</title>
<style>
  :root {
    --bg: #0b0d12; --panel: #12151c; --panel-2: #171b24; --border: #232836;
    --text: #e6e9f0; --muted: #8b92a5; --green: #23c58a; --red: #ef4c5f;
    --amber: #e5b13a; --accent: #5b8cff; --mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 14px; }
  a { color: var(--accent); }
  header { display: flex; align-items: center; gap: 12px; padding: 10px 16px;
    background: var(--panel); border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 10; }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; letter-spacing: 0.02em; }
  .badge { font-weight: 700; font-size: 12px; padding: 3px 10px; border-radius: 999px; letter-spacing: 0.05em; }
  .badge.paper { background: rgba(91,140,255,0.15); color: var(--accent); border: 1px solid rgba(91,140,255,0.4); }
  .badge.live { background: rgba(239,76,95,0.18); color: var(--red); border: 1px solid var(--red); animation: pulse 1.6s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.55; } }
  .spacer { flex: 1; }
  .conn-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); display: inline-block; margin-right: 6px; }
  .conn-dot.ok { background: var(--green); }
  .conn-dot.bad { background: var(--red); }
  #killBtn { background: var(--red); color: #fff; border: none; font-weight: 700; padding: 8px 16px;
    border-radius: 6px; cursor: pointer; letter-spacing: 0.04em; font-size: 12px; }
  #killBtn:hover { filter: brightness(1.15); }
  main { display: grid; grid-template-columns: 300px 1fr 320px; gap: 12px; padding: 12px; align-items: start; }
  @media (max-width: 1000px) { main { grid-template-columns: 1fr; } }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; margin-bottom: 12px; }
  .panel h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted);
    margin: 0 0 8px 0; font-weight: 600; }
  .kv { display: flex; justify-content: space-between; padding: 3px 0; }
  .kv .k { color: var(--muted); } .kv .v { font-family: var(--mono); }
  .v.pos { color: var(--green); } .v.neg { color: var(--red); }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; color: var(--muted); font-weight: 500; padding: 4px 4px; border-bottom: 1px solid var(--border); }
  td { padding: 4px 4px; font-family: var(--mono); border-bottom: 1px solid rgba(255,255,255,0.03); }
  .empty { color: var(--muted); font-style: italic; padding: 6px 0; font-size: 12px; }
  .side-buy { color: var(--green); } .side-sell { color: var(--red); }
  select, input[type=text], input[type=number] { background: var(--panel-2); border: 1px solid var(--border);
    color: var(--text); border-radius: 5px; padding: 6px 8px; font-size: 12px; width: 100%; }
  label { display: block; font-size: 11px; color: var(--muted); margin: 8px 0 3px; }
  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  button.action { width: 100%; padding: 9px; border-radius: 6px; border: none; font-weight: 700;
    cursor: pointer; margin-top: 12px; font-size: 12px; letter-spacing: 0.03em; }
  button.buy { background: var(--green); color: #06210f; }
  button.sell { background: var(--red); color: #2a0006; }
  .cancel-btn { background: transparent; border: 1px solid var(--border); color: var(--muted);
    border-radius: 4px; padding: 2px 8px; cursor: pointer; font-size: 11px; }
  .cancel-btn:hover { color: var(--red); border-color: var(--red); }
  #symbolTabs { display: flex; gap: 6px; margin-bottom: 10px; flex-wrap: wrap; }
  .tab { padding: 5px 12px; border-radius: 6px; background: var(--panel-2); color: var(--muted);
    cursor: pointer; font-size: 12px; border: 1px solid var(--border); }
  .tab.active { color: var(--text); border-color: var(--accent); background: rgba(91,140,255,0.1); }
  .book { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .book-col table { width: 100%; }
  .book-col th { font-size: 10px; }
  .depth-cell { position: relative; }
  .depth-bar { position: absolute; top: 0; bottom: 0; opacity: 0.18; z-index: 0; }
  .depth-bar.bid { background: var(--green); right: 0; }
  .depth-bar.ask { background: var(--red); left: 0; }
  .depth-cell span { position: relative; z-index: 1; }
  .imbalance-wrap { margin: 10px 0 4px; }
  .imbalance-bar { height: 8px; border-radius: 4px; background: var(--panel-2); overflow: hidden; display: flex; }
  .imbalance-bar .bid-fill { background: var(--green); } .imbalance-bar .ask-fill { background: var(--red); }
  .imbalance-label { display: flex; justify-content: space-between; font-size: 10px; color: var(--muted); margin-top: 3px; }
  .tape { max-height: 220px; overflow-y: auto; }
  .modal-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,0.6); display: none;
    align-items: center; justify-content: center; z-index: 100; }
  .modal-backdrop.show { display: flex; }
  .modal { background: var(--panel); border: 1px solid var(--red); border-radius: 10px; padding: 22px; width: 320px; text-align: center; }
  .modal h3 { color: var(--red); margin-top: 0; }
  .modal button { margin: 6px; padding: 8px 18px; border-radius: 6px; border: none; cursor: pointer; font-weight: 700; }
  .modal .confirm { background: var(--red); color: #fff; }
  .modal .cancel { background: var(--panel-2); color: var(--text); border: 1px solid var(--border); }
  .util-bar { height: 6px; border-radius: 3px; background: var(--panel-2); overflow: hidden; margin-top: 2px; }
  .util-bar > div { height: 100%; background: var(--accent); }
  .util-bar.danger > div { background: var(--red); }
  .toast { position: fixed; bottom: 16px; right: 16px; background: var(--panel-2); border: 1px solid var(--border);
    padding: 10px 16px; border-radius: 8px; font-size: 12px; z-index: 200; max-width: 320px; }
</style>
</head>
<body>
<header>
  <h1>JEVTRADER</h1>
  <span id="modeBadge" class="badge paper">PAPER</span>
  <span style="color: var(--muted); font-size: 12px;"><span id="connDot" class="conn-dot"></span><span id="connLabel">connecting</span></span>
  <div class="spacer"></div>
  <button id="killBtn" onclick="openKillModal()">KILL SWITCH</button>
</header>

<main>
  <!-- LEFT: strategies, open orders, manual ticket -->
  <div>
    <div class="panel">
      <h2>Account</h2>
      <div class="kv"><span class="k">Equity</span><span class="v" id="acctEquity">--</span></div>
      <div class="kv"><span class="k">Cash</span><span class="v" id="acctCash">--</span></div>
      <div class="kv"><span class="k">Buying power</span><span class="v" id="acctBP">--</span></div>
      <div class="kv"><span class="k">Drawdown</span><span class="v" id="acctDD">--</span></div>
    </div>

    <div class="panel">
      <h2>Risk</h2>
      <div class="kv"><span class="k">Status</span><span class="v" id="riskStatus">--</span></div>
      <div id="riskUtil"></div>
    </div>

    <div class="panel">
      <h2>Strategies</h2>
      <table><thead><tr><th>id</th><th>status</th><th>pnl</th><th>pos</th></tr></thead>
        <tbody id="strategyBody"><tr><td colspan="4" class="empty">no strategies</td></tr></tbody>
      </table>
    </div>

    <div class="panel">
      <h2>Manual order ticket (via RiskGate)</h2>
      <label>Symbol</label>
      <input type="text" id="ordSymbol" placeholder="AAPL or BTC/USD">
      <div class="row2">
        <div><label>Type</label>
          <select id="ordType">
            <option value="market">market</option>
            <option value="limit">limit</option>
            <option value="stop">stop</option>
            <option value="stop_limit">stop_limit</option>
          </select>
        </div>
        <div><label>Qty</label><input type="number" id="ordQty" step="any" value="1"></div>
      </div>
      <div class="row2">
        <div><label>Limit price</label><input type="number" id="ordLimit" step="any"></div>
        <div><label>Stop price</label><input type="number" id="ordStop" step="any"></div>
      </div>
      <label style="display:flex; align-items:center; gap:6px; margin-top:10px;">
        <input type="checkbox" id="ordPostOnly" style="width:auto;"> post-only
      </label>
      <div class="row2">
        <button class="action buy" onclick="submitOrder('buy')">BUY</button>
        <button class="action sell" onclick="submitOrder('sell')">SELL</button>
      </div>
    </div>
  </div>

  <!-- CENTER: order book + tape -->
  <div>
    <div class="panel">
      <h2>Order book</h2>
      <div id="symbolTabs"></div>
      <div class="imbalance-wrap">
        <div class="imbalance-bar"><div class="bid-fill" id="imbBid" style="width:50%"></div><div class="ask-fill" id="imbAsk" style="width:50%"></div></div>
        <div class="imbalance-label"><span id="imbBidLabel">bid 0%</span><span id="imbAskLabel">ask 0%</span></div>
      </div>
      <div class="book">
        <div class="book-col">
          <table><thead><tr><th>bid size</th><th>price</th></tr></thead><tbody id="bidBody"></tbody></table>
        </div>
        <div class="book-col">
          <table><thead><tr><th>price</th><th>ask size</th></tr></thead><tbody id="askBody"></tbody></table>
        </div>
      </div>
    </div>

    <div class="panel">
      <h2>Trades tape</h2>
      <div class="tape">
        <table><thead><tr><th>time</th><th>side</th><th>price</th><th>size</th></tr></thead>
          <tbody id="tapeBody"><tr><td colspan="4" class="empty">no trades yet</td></tr></tbody>
        </table>
      </div>
    </div>

    <div class="panel">
      <h2>Open orders</h2>
      <table><thead><tr><th>symbol</th><th>side</th><th>type</th><th>qty</th><th>status</th><th></th></tr></thead>
        <tbody id="ordersBody"><tr><td colspan="6" class="empty">no open orders</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- RIGHT: positions + jev feed -->
  <div>
    <div class="panel">
      <h2>Positions</h2>
      <table><thead><tr><th>symbol</th><th>qty</th><th>avg</th><th>upnl</th></tr></thead>
        <tbody id="positionsBody"><tr><td colspan="4" class="empty">flat</td></tr></tbody>
      </table>
    </div>

    <div class="panel">
      <h2>Jev decision feed</h2>
      <div class="tape" id="jevFeed"><div class="empty">no decisions yet</div></div>
    </div>
  </div>
</main>

<div class="modal-backdrop" id="killModal">
  <div class="modal">
    <h3>KILL SWITCH</h3>
    <p>This cancels every open order and flattens every position, immediately.</p>
    <div>
      <button class="cancel" onclick="closeKillModal()">Cancel</button>
      <button class="confirm" onclick="confirmKill()">Confirm kill</button>
    </div>
  </div>
</div>

<script>
let state = null;
let activeSymbol = null;
let ws = null;

function fmt(n, d) { if (n === undefined || n === null || Number.isNaN(n)) return "--";
  return Number(n).toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d}); }
function pct(n) { return n === undefined || n === null ? "--" : (n * 100).toFixed(2) + "%"; }
function esc(s) { return String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(proto + "://" + location.host + "/ws");
  ws.onopen = () => setConn(true);
  ws.onclose = () => { setConn(false); setTimeout(connect, 1500); };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => render(JSON.parse(ev.data));
}
function setConn(ok) {
  document.getElementById("connDot").className = "conn-dot " + (ok ? "ok" : "bad");
  document.getElementById("connLabel").textContent = ok ? "live" : "reconnecting";
}

async function pollOnce() {
  try { const r = await fetch("/api/snapshot"); render(await r.json()); } catch (e) {}
}

function render(s) {
  state = s;
  const badge = document.getElementById("modeBadge");
  badge.textContent = s.mode;
  badge.className = "badge " + (s.mode === "LIVE" ? "live" : "paper");

  document.getElementById("acctEquity").textContent = fmt(s.account.equity, 2);
  document.getElementById("acctCash").textContent = fmt(s.account.cash, 2);
  document.getElementById("acctBP").textContent = fmt(s.account.buying_power, 2);
  const dd = document.getElementById("acctDD");
  dd.textContent = pct(s.account.drawdown);
  dd.className = "v " + (s.account.drawdown > 0.1 ? "neg" : "");

  const riskStatus = document.getElementById("riskStatus");
  riskStatus.textContent = s.risk.halted ? "HALTED" : "active";
  riskStatus.className = "v " + (s.risk.halted ? "neg" : "pos");
  const util = document.getElementById("riskUtil");
  util.innerHTML = Object.entries(s.risk.utilization || {}).map(([k, v]) => {
    const p = Math.min(1, Number(v));
    return `<div class="kv"><span class="k">${esc(k)}</span><span class="v">${pct(p)}</span></div>
            <div class="util-bar ${p > 0.85 ? "danger" : ""}"><div style="width:${p * 100}%"></div></div>`;
  }).join("") || "";

  renderTable("strategyBody", Object.values(s.strategies), 4, (st) =>
    `<td>${esc(st.strategy_id || "")}</td><td>${esc(st.status || "")}</td>
     <td class="${(st.pnl || 0) >= 0 ? "" : "neg"}">${fmt(st.pnl, 2)}</td><td>${fmt(st.position, 4)}</td>`);

  renderTable("positionsBody", Object.values(s.positions).filter(p => Math.abs(p.qty || 0) > 1e-9), 4, (p) =>
    `<td>${esc(p.symbol)}</td><td>${fmt(p.qty, 4)}</td><td>${fmt(p.avg_price, 2)}</td>
     <td class="${(p.unrealized_pnl || 0) >= 0 ? "" : "neg"}">${fmt(p.unrealized_pnl, 2)}</td>`, "flat");

  renderTable("ordersBody", Object.values(s.open_orders), 6, (o) =>
    `<td>${esc(o.symbol)}</td><td class="side-${o.side}">${esc(o.side)}</td><td>${esc(o.type)}</td>
     <td>${fmt(o.qty, 4)}</td><td>${esc(o.status)}</td>
     <td><button class="cancel-btn" onclick="cancelOrder('${esc(o.client_order_id)}')">cancel</button></td>`);

  renderJevFeed(s.jev_feed);
  renderSymbolTabs(Object.keys(s.books));
  renderBook();
  renderTape();
}

function renderTable(id, rows, cols, rowFn, emptyMsg) {
  const body = document.getElementById(id);
  if (!rows.length) { body.innerHTML = `<tr><td colspan="${cols}" class="empty">${emptyMsg || "none"}</td></tr>`; return; }
  body.innerHTML = rows.map(r => `<tr>${rowFn(r)}</tr>`).join("");
}

function renderJevFeed(feed) {
  const el = document.getElementById("jevFeed");
  if (!feed || !feed.length) { el.innerHTML = '<div class="empty">no decisions yet</div>'; return; }
  el.innerHTML = feed.slice(0, 30).map(d => {
    const top = Object.entries(d.top || {}).map(([q, v]) => `${esc(q)}=${esc(v)}`).join(", ");
    const late = d.late ? ' <span class="neg">LATE</span>' : "";
    return `<div class="kv"><span class="k">${esc(d.symbol)} · ${esc(d.question)}</span>
            <span class="v">${top} (${fmt(d.latency_ms, 0)}ms)${late}</span></div>`;
  }).join("");
}

function renderSymbolTabs(symbols) {
  symbols = symbols.sort();
  if (!activeSymbol || !symbols.includes(activeSymbol)) activeSymbol = symbols[0] || null;
  const tabs = document.getElementById("symbolTabs");
  tabs.innerHTML = symbols.map(s =>
    `<div class="tab ${s === activeSymbol ? "active" : ""}" onclick="selectSymbol('${esc(s)}')">${esc(s)}</div>`
  ).join("") || '<span class="empty">no symbols streaming</span>';
}
function selectSymbol(s) { activeSymbol = s; renderBook(); renderTape(); }

function renderBook() {
  const bidBody = document.getElementById("bidBody"), askBody = document.getElementById("askBody");
  const book = activeSymbol && state.books[activeSymbol];
  if (!book) { bidBody.innerHTML = askBody.innerHTML = '<tr><td colspan="2" class="empty">no book</td></tr>'; setImbalance(0, 0); return; }
  const maxSize = Math.max(1e-9, ...book.bids.map(l => l.size), ...book.asks.map(l => l.size));
  bidBody.innerHTML = book.bids.map(l =>
    `<tr><td class="depth-cell"><div class="depth-bar bid" style="width:${(l.size / maxSize) * 100}%"></div><span>${fmt(l.size, 4)}</span></td>
     <td class="side-buy">${fmt(l.price, 4)}</td></tr>`).join("") || '<tr><td colspan="2" class="empty">--</td></tr>';
  askBody.innerHTML = book.asks.map(l =>
    `<tr><td class="side-sell">${fmt(l.price, 4)}</td>
     <td class="depth-cell"><div class="depth-bar ask" style="width:${(l.size / maxSize) * 100}%"></div><span>${fmt(l.size, 4)}</span></td></tr>`).join("") || '<tr><td colspan="2" class="empty">--</td></tr>';
  const bidVol = book.bids.reduce((a, l) => a + l.size, 0), askVol = book.asks.reduce((a, l) => a + l.size, 0);
  const tot = bidVol + askVol;
  setImbalance(tot > 0 ? bidVol / tot : 0.5, tot > 0 ? askVol / tot : 0.5);
}
function setImbalance(bidFrac, askFrac) {
  document.getElementById("imbBid").style.width = (bidFrac * 100) + "%";
  document.getElementById("imbAsk").style.width = (askFrac * 100) + "%";
  document.getElementById("imbBidLabel").textContent = "bid " + Math.round(bidFrac * 100) + "%";
  document.getElementById("imbAskLabel").textContent = "ask " + Math.round(askFrac * 100) + "%";
}

function renderTape() {
  const body = document.getElementById("tapeBody");
  const trades = (activeSymbol && state.trades[activeSymbol]) || [];
  if (!trades.length) { body.innerHTML = '<tr><td colspan="4" class="empty">no trades yet</td></tr>'; return; }
  body.innerHTML = trades.slice(0, 50).map(t =>
    `<tr><td>${new Date(t.ts).toLocaleTimeString()}</td><td class="side-${t.side || ""}">${esc(t.side || "")}</td>
     <td>${fmt(t.price, 4)}</td><td>${fmt(t.size, 4)}</td></tr>`).join("");
}

async function submitOrder(side) {
  const payload = {
    symbol: document.getElementById("ordSymbol").value.trim(),
    side, type: document.getElementById("ordType").value,
    qty: parseFloat(document.getElementById("ordQty").value),
    limit_price: parseFloat(document.getElementById("ordLimit").value) || null,
    stop_price: parseFloat(document.getElementById("ordStop").value) || null,
    post_only: document.getElementById("ordPostOnly").checked,
  };
  if (!payload.symbol || !payload.qty) { toast("symbol and qty are required"); return; }
  try {
    const r = await fetch("/api/orders", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    const body = await r.json();
    toast(r.ok ? `order submitted (${body.client_order_id})` : `rejected: ${body.detail || body.reason || "unknown"}`);
  } catch (e) { toast("order submit failed: " + e); }
}

async function cancelOrder(cid) {
  try { await fetch("/api/orders/" + encodeURIComponent(cid), {method: "DELETE"}); } catch (e) { toast("cancel failed: " + e); }
}

function openKillModal() { document.getElementById("killModal").classList.add("show"); }
function closeKillModal() { document.getElementById("killModal").classList.remove("show"); }
async function confirmKill() {
  closeKillModal();
  try {
    const r = await fetch("/api/kill", {method: "POST"});
    const body = await r.json();
    toast(body.ok ? "kill switch engaged" : `kill switch failed: ${body.reason || "unknown"}`);
  } catch (e) { toast("kill switch failed: " + e); }
}

let toastTimer = null;
function toast(msg) {
  let el = document.getElementById("toastEl");
  if (!el) { el = document.createElement("div"); el.id = "toastEl"; el.className = "toast"; document.body.appendChild(el); }
  el.textContent = msg;
  el.style.display = "block";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.style.display = "none"; }, 4000);
}

connect();
pollOnce();
setInterval(() => { if (!ws || ws.readyState !== 1) pollOnce(); }, 2000);
</script>
</body>
</html>
"""
