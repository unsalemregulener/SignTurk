// Shared basics: API access, session, theme, formatting, icons and the brand marks.
// Every file is a classic script compiled by Babel in the browser, so top-level
// names are shared between files and must be unique.
const {useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState} = React;

const ORIGIN = window.location.origin;
const WS_BASE = (window.location.protocol === "https:" ? "wss://" : "ws://") + window.location.host;
const API = {
  health: "/api/health",
  login: "/api/login",
  logout: "/api/logout",
  register: "/api/register",
  live: "/api/predict/live226",
  history: "/api/history",
  summary: "/api/me/summary",
  dictionary: "/api/dictionary",
  settings: "/api/settings",
  adminOverview: "/api/admin/overview",
  adminUsers: "/api/admin/users",
  adminLogs: "/api/admin/logs",
  adminModel: "/api/admin/model",
  avatar: "/avatar3d"
};
const KEYS = {user:"signturk_user", theme:"signturk_theme", setup:"signturk_setup_done", camera:"signturk_camera"};
const RECORD_FRAMES = 48;
const MODEL_STATES = {ready:"Hazır", loading:"Yükleniyor", idle:"Bekliyor", missing:"Dosyalar eksik", error:"Hata", disabled:"Kapalı"};
const DEVICE_NAMES = {cuda:"GPU", cpu:"CPU"};
const TOUCH = window.matchMedia("(hover: none)").matches;

function readKey(key){ try{ return localStorage.getItem(key); }catch{ return null; } }
function writeKey(key, value){ try{ value == null ? localStorage.removeItem(key) : localStorage.setItem(key, value); }catch{} }
function readJSON(key){ try{ return JSON.parse(readKey(key) || "null"); }catch{ return null; } }
function storedToken(){ return readJSON(KEYS.user)?.access_token || ""; }

function errorMessage(data){
  if (typeof data?.detail === "string") return data.detail;
  if (Array.isArray(data?.detail) && data.detail[0]?.msg) return data.detail[0].msg;
  return data?.error || "İstek tamamlanamadı";
}

async function apiFetch(path, options={}){
  const token = storedToken();
  const res = await fetch(ORIGIN + path, {
    ...options,
    headers:{"Content-Type":"application/json", ...(token ? {Authorization:`Bearer ${token}`} : {}), ...(options.headers || {})}
  });
  if (res.status === 401 && token) window.dispatchEvent(new Event("signturk:logout"));
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) throw new Error(errorMessage(data));
  return data;
}

// ── Formatting ───────────────────────────────────────────
function percent(value){ return `%${Math.round(Math.max(0, Math.min(1, Number(value) || 0)) * 100)}`; }

// "59.8" or "59.8%" from the server, written the Turkish way: %59,8
function trPercent(value){
  const number = Number(String(value ?? "").replace("%", ""));
  if (value == null || value === "-" || value === "" || Number.isNaN(number)) return "-";
  return `%${number.toLocaleString("tr-TR", {maximumFractionDigits:1})}`;
}
function confidenceOf(value){
  const number = Number(String(value ?? "").replace("%", ""));
  return Number.isNaN(number) ? null : number / 100;
}
function formatTime(iso, withDate=false){
  if (!iso) return "-";
  const date = new Date(iso);
  return withDate
    ? date.toLocaleString("tr-TR", {day:"numeric", month:"long", hour:"2-digit", minute:"2-digit"})
    : date.toLocaleTimeString("tr-TR", {hour:"2-digit", minute:"2-digit"});
}
function dayKey(date){ return `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`; }
function dayLabel(iso){
  const date = new Date(iso);
  const today = new Date();
  const yesterday = new Date(Date.now() - 86400000);
  if (dayKey(date) === dayKey(today)) return "Bugün";
  if (dayKey(date) === dayKey(yesterday)) return "Dün";
  return date.toLocaleDateString("tr-TR", {day:"numeric", month:"long", year:"numeric"});
}
function initials(user){
  const source = user?.full_name || user?.name || user?.email || "ST";
  return source.split(/[\s@.]+/).filter(Boolean).slice(0, 2).map(x => x[0]?.toLocaleUpperCase("tr-TR")).join("") || "ST";
}
function firstName(user){ return (user?.full_name || "").trim().split(/\s+/)[0] || ""; }
function upperTR(text){ return String(text || "").toLocaleUpperCase("tr-TR"); }

// Words approved on each of the last `days` days, oldest first.
function dailyCounts(rows, days){
  const counts = new Map();
  for (const row of rows){
    if (!row.created_at) continue;
    const key = dayKey(new Date(row.created_at));
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  const out = [];
  for (let i = days - 1; i >= 0; i--){
    const date = new Date(Date.now() - i * 86400000);
    out.push({date, count:counts.get(dayKey(date)) || 0, today:i === 0});
  }
  return out;
}

function speak(text, rate=1){
  if (!text || !window.speechSynthesis) return;
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = "tr-TR";
  utter.rate = rate;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utter);
}

// ── Theme: follows the system unless the user picked one ──
function themePref(){ const value = readKey(KEYS.theme); return value === "light" || value === "dark" ? value : "system"; }
function applyTheme(pref){
  if (pref === "light" || pref === "dark") document.documentElement.dataset.theme = pref;
  else delete document.documentElement.dataset.theme;
  document.querySelectorAll("iframe.avatar-frame").forEach(frame => frame.contentWindow?.postMessage({type:"set_theme", value:pref}, ORIGIN));
}
applyTheme(themePref());

function useTheme(){
  const [pref, setPref] = useState(themePref);
  const choose = useCallback((next) => { writeKey(KEYS.theme, next === "system" ? null : next); applyTheme(next); setPref(next); }, []);
  return [pref, choose];
}

// The model's state from /api/health, polled while it loads.
function useHealth(){
  const [model, setModel] = useState(undefined);
  useEffect(() => {
    let alive = true, timer;
    const load = () => fetch(ORIGIN + API.health).then(r => r.json())
      .then(h => { if (alive) setModel(h.model_226 || null); return h.model_226; })
      .catch(() => { if (alive) setModel(null); })
      .then(m => { if (alive) timer = setTimeout(load, m?.ready ? 30000 : 3000); });
    load();
    return () => { alive = false; clearTimeout(timer); };
  }, []);
  return model;
}

// ── Icons (Phosphor, see icons.js) ──────────────────────
function Icon({name, size=20, duo=false, className="", title}){
  const icon = window.PHOSPHOR?.[name];
  if (!icon) return <span className="icon" style={{width:size, height:size}}/>;
  const markup = (duo ? icon[1].replace(/<(\w+)/g, '<$1 class="tint"') : "") + icon[0];
  return (
    <svg className={`icon ${duo ? "duo" : ""} ${className}`} width={size} height={size} viewBox="0 0 256 256" fill="currentColor"
      aria-hidden={title ? undefined : "true"} role={title ? "img" : undefined} dangerouslySetInnerHTML={{__html:(title ? `<title>${title}</title>` : "") + markup}}/>
  );
}

// ── Brand ───────────────────────────────────────────────
let svgIds = 0;
function useSvgId(prefix){ const ref = useRef(null); if (!ref.current) ref.current = `${prefix}${++svgIds}`; return ref.current; }

// The mark: two hands, cobalt and amber, each raising two fingers, signing an S
// together. Stroked centrelines drawn bottom to top; the order makes the weave.
// The same paths make brand/*.svg.
const MARK_PATHS = [
  ["b", "M270 205C330 205 385 214 400 245C408 262 405 278 400 288"],
  ["a", "M300 178C360 176 430 170 470 180C530 195 560 240 552 288"],
  ["a", "M368 97C330 135 265 190 240 225C215 262 215 330 245 370C275 405 330 420 365 438C395 455 410 480 410 520C412 560 405 600 380 630C340 675 290 690 240 690C170 690 115 655 85 598"],
  ["b", "M135 97C110 155 85 230 75 330"],
  ["b", "M265 65C215 125 160 180 120 235C88 280 70 320 70 370C70 460 140 548 240 548C330 548 385 470 385 362C470 358 558 430 562 540C566 650 480 712 390 712C350 712 310 702 285 688"],
  ["a", "M380 630C340 675 290 690 240 690"]
];
const MARK_VIEW = "36 31 560 715";

function MarkPaths(){
  return <g strokeWidth="66" strokeLinecap="round" strokeLinejoin="round">{MARK_PATHS.map(([hand, d], i) => <path key={i} className={`mark-${hand}`} d={d}/>)}</g>;
}

function Mark({size=40}){
  return <svg className="mark" width={Math.round(size * 560 / 715)} height={size} viewBox={MARK_VIEW} fill="none" aria-hidden="true"><MarkPaths/></svg>;
}

function Logo({size=34, word=true}){
  return <span className="logo"><Mark size={size}/>{word && <span className="logo-word">SignTurk</span>}</span>;
}

// The mark again, large and faint, as a watermark on the dark brand panels.
function BrandArt({className=""}){
  return <svg className={`brand-art ${className}`} viewBox={MARK_VIEW} fill="none" aria-hidden="true"><MarkPaths/></svg>;
}
