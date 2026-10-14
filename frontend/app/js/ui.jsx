// Shared interface pieces built on app.css.

// Overlays render into <body>, above the navigation rail and outside the page's stacking context.
function Portal({children}){ return ReactDOM.createPortal(children, document.body); }

function PageHead({title, lead, eyebrow, children}){
  return (
    <header className="page-head rise">
      <div className="titles">
        {eyebrow && <div className="eyebrow" style={{marginBottom:10}}>{eyebrow}</div>}
        <h1>{title}</h1>
        {lead && <p className="lead">{lead}</p>}
      </div>
      {children && <div className="actions">{children}</div>}
    </header>
  );
}

function Seg({options, value, onChange, label, glassy=false, disabled=false}){
  return (
    <div className={`seg ${glassy ? "glassy" : ""}`} role="group" aria-label={label}>
      {options.map(o => (
        <button key={o.value} type="button" aria-pressed={value === o.value} disabled={disabled || o.disabled} title={o.title}
          onClick={() => value !== o.value && onChange(o.value)}>
          {o.icon && <Icon name={o.icon} size={17} duo={value === o.value}/>}{o.label}
        </button>
      ))}
    </div>
  );
}

function Switch({checked, onChange, label, glass=false}){
  return <button type="button" role="switch" aria-checked={!!checked} aria-label={label} className={`switch ${glass ? "on-glass" : ""}`} onClick={() => onChange(!checked)}/>;
}

function Range({value, min, max, step, onChange, label}){
  const p = ((value - min) / (max - min)) * 100;
  return <input className="range" type="range" min={min} max={max} step={step} value={value} aria-label={label}
    style={{"--p":`${p}%`}} onChange={e => onChange(Number(e.target.value))}/>;
}

function Meter({value, kind="", thin=false, delay=0}){
  const width = Math.max(2, Math.min(100, Math.round((Number(value) || 0) * 100)));
  return <div className={`meter ${kind} ${thin ? "thin" : ""}`} aria-hidden="true"><i style={{width:`${width}%`, animationDelay:`${delay}ms`}}/></div>;
}

function StatusDot({tone, children}){
  return <span className="status"><span className={`dot ${tone || ""}`}/>{children}</span>;
}

function ModelStatus({model}){
  if (model === undefined) return <StatusDot>Bağlanıyor</StatusDot>;
  if (!model) return <StatusDot tone="bad">Sunucuya ulaşılamıyor</StatusDot>;
  if (model.ready) return <StatusDot tone="good">Model hazır · {DEVICE_NAMES[model.device] || "-"}</StatusDot>;
  if (model.state === "loading" || model.state === "idle") return <StatusDot tone="warn">Model yükleniyor</StatusDot>;
  return <StatusDot tone="bad">{MODEL_STATES[model.state] || "Model hazır değil"}</StatusDot>;
}

function Empty({icon="sparkle", image, title, children, action}){
  return (
    <div className="empty rise">
      {image ? <img src={image} alt="" onError={e => { e.currentTarget.style.display = "none"; }}/> : <div className="empty-icon"><Icon name={icon} size={30} duo/></div>}
      {title && <h3>{title}</h3>}
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

function Skeleton({height=20, width="100%", style}){
  return <div className="skeleton" style={{height, width, ...style}}/>;
}

// One toast at a time: useToast() returns [node, show(text, tone)].
function useToast(){
  const [toast, setToast] = useState(null);
  const timer = useRef(null);
  const show = useCallback((text, tone="good") => {
    clearTimeout(timer.current);
    setToast({text, tone, key:Date.now()});
    timer.current = setTimeout(() => setToast(null), 2200);
  }, []);
  useEffect(() => () => clearTimeout(timer.current), []);
  const node = toast ? (
    <Portal>
      <div key={toast.key} className="toast" role="status">
        <Icon name={toast.tone === "bad" ? "warning-circle" : "check-circle"} size={20} duo/>{toast.text}
      </div>
    </Portal>
  ) : null;
  return [node, show];
}

// The 3D avatar page in a frame; plays `sign` whenever it changes.
function AvatarFrame({sign, className="", style}){
  const frame = useRef(null);
  const speed = useRef(1);
  const send = useCallback(() => {
    const target = frame.current?.contentWindow;
    if (!target) return;
    target.postMessage({type:"set_theme", value:themePref()}, ORIGIN);
    target.postMessage({type:"set_speed", value:speed.current}, ORIGIN);
    if (sign?.gloss) target.postMessage({type:"play_sign", word:sign.gloss, label:sign.text || sign.gloss}, ORIGIN);
  }, [sign?.gloss, sign?.text]);
  useEffect(() => { apiFetch(API.settings).then(s => { speed.current = Number(s?.avatar_speed) || 1; }).catch(() => {}); }, []);
  useEffect(() => { send(); }, [send]);
  return <iframe ref={frame} className={`avatar-frame ${className}`} style={style} src={API.avatar} title={`${sign?.text || "İşaret"} animasyonu`} onLoad={send}/>;
}

// "How is it signed?": the avatar in a side drawer.
function HowToDrawer({sign, onClose}){
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <Portal>
    <div className="drawer-scrim" onClick={onClose}>
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={`${sign.text} nasıl yapılır`} onClick={e => e.stopPropagation()}>
        <div className="drawer-head">
          <div><div className="eyebrow"><Icon name="hand-waving" size={16} duo/>Nasıl yapılır</div><h2 style={{marginTop:6}}>{sign.text}</h2></div>
          <span className="spacer"/>
          <button className="btn btn-ghost icon-only" onClick={onClose} aria-label="Kapat"><Icon name="x" size={20}/></button>
        </div>
        <AvatarFrame sign={sign}/>
        <div className="drawer-foot"><Icon name="info" size={18}/>Animasyon AUTSL kayıtlarından çıkarıldı. Tekrar, yavaş ve döngü düğmeleri görüntünün sol altında.</div>
      </aside>
    </div>
    </Portal>
  );
}

// Images of the avatar with a transparent background, made from the avatar page.
const AVATAR_POSES = {
  wave:"/frontend/app/brand/avatar-merhaba.webp",
  yes:"/frontend/app/brand/avatar-evet.webp",
  rest:"/frontend/app/brand/avatar-bekle.webp"
};
