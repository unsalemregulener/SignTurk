// The camera and the live socket: framing previews between recordings, a
// 4-second recording on request, or the hands-free stream.

function useLive(model, onExpired){
  const video = useRef(null);
  const canvas = useRef(null);
  const socket = useRef(null);
  const media = useRef(null);
  const timers = useRef({});
  const previewBusy = useRef(false);
  const inFlight = useRef(0);
  const modeRef = useRef("record");
  const phaseRef = useRef("idle");
  const autoRef = useRef(false);
  const okRun = useRef(0);
  const listener = useRef(null);
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [framing, setFraming] = useState(null);
  const [steady, setSteady] = useState(false);
  const [phase, setPhaseState] = useState("idle");   // idle, countdown, recording, reading
  const [count, setCount] = useState(0);
  const [progress, setProgress] = useState(0);
  const [watch, setWatch] = useState("off");         // off, waiting, signing, reading
  const [mode, setModeState] = useState("record");   // record, continuous
  const [auto, setAutoState] = useState(false);
  const [aspect, setAspect] = useState(16 / 10);
  const [devices, setDevices] = useState([]);
  const [deviceId, setDeviceId] = useState(() => readKey(KEYS.camera) || "");
  const settings = useRef({});
  settings.current = {
    frames: model?.record_frames || RECORD_FRAMES,
    frameMs: Math.round(1000 / (model?.camera_fps || 12)),
    previewMs: model?.device === "cuda" ? 250 : 1000
  };

  const setPhase = (next) => { phaseRef.current = next; setPhaseState(next); };
  const clear = (name) => { const t = timers.current[name]; if (t){ clearInterval(t); clearTimeout(t); timers.current[name] = null; } };
  const clearAll = () => { Object.keys(timers.current).forEach(clear); previewBusy.current = false; inFlight.current = 0; };
  const send = (obj) => { const ws = socket.current; if (ws && ws.readyState === WebSocket.OPEN){ ws.send(JSON.stringify(obj)); return true; } return false; };

  // Any <video> that mounts gets the running camera, so screens can swap.
  const videoRef = useCallback((el) => {
    video.current = el;
    if (!el) return;
    el.onloadedmetadata = () => { if (el.videoWidth && el.videoHeight) setAspect(el.videoWidth / el.videoHeight); };
    if (media.current && el.srcObject !== media.current){ el.srcObject = media.current; el.play().catch(() => {}); }
  }, []);

  // Frames go out unmirrored, 640 px wide; only the preview is mirrored.
  const capture = () => {
    const v = video.current;
    if (!v || !v.videoWidth || !v.videoHeight) return null;
    if (!canvas.current) canvas.current = document.createElement("canvas");
    const c = canvas.current;
    c.width = 640; c.height = Math.round(640 * v.videoHeight / v.videoWidth);
    c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.8);
  };

  const showFraming = (f) => {
    setFraming(f);
    okRun.current = f?.ok ? okRun.current + 1 : 0;
    setSteady(okRun.current >= 3);
  };

  const preview = () => {
    clear("preview"); previewBusy.current = false;
    timers.current.preview = setInterval(() => {
      if (previewBusy.current) return;
      const image = capture();
      if (image && send({image, preview:true})) previewBusy.current = true;
    }, settings.current.previewMs);
  };

  // Hands-free: frames go out without a button, at most two awaiting an answer.
  const stream = () => {
    clear("preview"); clear("stream"); inFlight.current = 0;
    send({action:"continuous", auto_approve:autoRef.current});
    setWatch("waiting");
    timers.current.stream = setInterval(() => {
      if (inFlight.current >= 2) return;
      const image = capture();
      if (image && send({image})) inFlight.current += 1;
    }, settings.current.frameMs);
  };
  const stopStream = () => { clear("stream"); inFlight.current = 0; setWatch("off"); };

  const onMessage = (event) => {
    let data;
    try{ data = JSON.parse(event.data); }catch{ return; }
    if (data.type === "framing"){ previewBusy.current = false; showFraming(data); return; }
    if (data.type === "watching"){
      inFlight.current = Math.max(0, inFlight.current - 1);
      if (modeRef.current === "continuous"){ setWatch(data.state); if (data.framing) showFraming(data.framing); }
      return;
    }
    if (data.type === "unavailable"){ setError(data.message || "Model henüz hazır değil."); return; }
    if (data.type === "continuous"){
      if (!data.on && modeRef.current === "continuous"){
        stopStream(); modeRef.current = "record"; setModeState("record"); preview();
        if (data.error) setError(data.error);
      }
      return;
    }
    if (data.type === "result"){
      clear("reading");
      if (modeRef.current === "record"){ setPhase("idle"); setProgress(0); preview(); }
      else setWatch("waiting");
      if (data.framing) showFraming(data.framing);
    }
    listener.current?.(data);
  };

  const listDevices = async () => {
    try{ setDevices((await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "videoinput" && d.deviceId)); }catch{}
  };

  const openCamera = async (id) => {
    const constraints = (exact) => ({audio:false, video:{...(exact ? {deviceId:{exact}} : {}), width:{ideal:960}, height:{ideal:600}, frameRate:{ideal:15, max:15}}});
    let next;
    try{ next = await navigator.mediaDevices.getUserMedia(constraints(id)); }
    catch(err){ if (!id || err?.name === "NotAllowedError") throw err; next = await navigator.mediaDevices.getUserMedia(constraints("")); }
    media.current?.getTracks().forEach(t => t.stop());
    media.current = next;
    const s = next.getVideoTracks()[0]?.getSettings?.() || {};
    if (s.width && s.height) setAspect(s.width / s.height);
    const v = video.current;
    if (v){ v.srcObject = next; await v.play().catch(() => {}); }
    listDevices();
  };

  const stop = () => {
    clearAll();
    const ws = socket.current; socket.current = null;
    if (ws){ ws.onclose = null; ws.onmessage = null; try{ ws.close(1000); }catch{} }
    media.current?.getTracks().forEach(t => t.stop()); media.current = null;
    if (video.current) video.current.srcObject = null;
    okRun.current = 0;
    setRunning(false); setStarting(false); setPhase("idle"); setProgress(0); setWatch("off"); setFraming(null); setSteady(false);
  };

  const start = async () => {
    if (socket.current) return true;
    setStarting(true); setError("");
    try{ await openCamera(deviceId); }
    catch(err){
      setStarting(false);
      setError(err?.name === "NotAllowedError" ? "Kamera izni verilmedi. Adres çubuğundaki kamera simgesinden izin verin."
        : err?.name === "NotFoundError" ? "Bu cihazda kamera bulunamadı." : "Kamera açılamadı; başka bir uygulama kullanıyor olabilir.");
      return false;
    }
    return new Promise((resolve) => {
      const ws = new WebSocket(WS_BASE + API.live);
      socket.current = ws;
      ws.onopen = () => {
        // The token goes in the first message, not in the URL, so it stays out of server logs.
        ws.send(JSON.stringify({action:"auth", token:storedToken()}));
        setRunning(true); setStarting(false);
        if (modeRef.current === "continuous") stream(); else preview();
        resolve(true);
      };
      ws.onmessage = onMessage;
      ws.onclose = (event) => {
        const expired = event.code === 1008;
        stop();
        if (expired){ setError("Oturumunuzun süresi doldu; yeniden giriş yapın."); onExpired?.(); }
        else if (event.code !== 1013) setError("Sunucuyla bağlantı kesildi.");
        resolve(false);
      };
    });
  };

  const record = () => {
    if (phaseRef.current !== "idle" || modeRef.current !== "record" || !socket.current) return false;
    clear("preview");
    setPhase("countdown"); setProgress(0);
    let n = 3; setCount(n);
    const tick = () => {
      n -= 1;
      if (n > 0){ setCount(n); timers.current.count = setTimeout(tick, 1000); return; }
      setPhase("recording");
      send({action:"start"});
      let sent = 0;
      const {frames: total, frameMs: every} = settings.current;
      timers.current.rec = setInterval(() => {
        const image = capture();
        if (image && send({image})){ sent += 1; setProgress(sent / total); }
        if (sent >= total){
          clear("rec"); send({action:"end"}); setPhase("reading");
          timers.current.reading = setTimeout(() => { setPhase("idle"); setError("Sonuç gelmedi; bir kez daha deneyin."); preview(); }, 20000);
        }
      }, every);
    };
    timers.current.count = setTimeout(tick, 1000);
    return true;
  };

  const setMode = (next) => {
    if (next === modeRef.current || phaseRef.current !== "idle") return;
    modeRef.current = next; setModeState(next); setError("");
    if (!socket.current) return;
    if (next === "continuous") stream();
    else{ stopStream(); send({action:"continuous", on:false}); preview(); }
  };

  const setAuto = (next) => {
    autoRef.current = next; setAutoState(next);
    if (timers.current.stream) send({action:"continuous", auto_approve:next});
  };

  const chooseDevice = async (id) => {
    setDeviceId(id); writeKey(KEYS.camera, id || null);
    if (!media.current) return;
    try{ await openCamera(id); }catch{ setError("Bu kamera açılamadı."); }
  };

  useEffect(() => { listDevices(); return () => stop(); }, []);
  return {videoRef, running, starting, error, setError, framing, steady, phase, count, progress, watch, mode, setMode,
          auto, setAuto, aspect, devices, deviceId, chooseDevice, start, stop, record, send, listener};
}

function framingLine(live){
  if (!live.framing) return {text:"Kadraj ölçülüyor", tone:""};
  return live.framing.ok ? {text:"Kadraj uygun", tone:"good"} : {text:live.framing.hint, tone:"bad"};
}

// The outline of a signer framed like AUTSL: shoulders about a fifth of the
// frame width across and a third of the way down, twice their width free below.
function Guide({state, aspect, faint}){
  const h = 160 / aspect;
  const dy = (h - 100) * 0.3;
  const ok = state === "ok";
  return (
    <svg className="guide" viewBox={`0 0 160 ${h}`} preserveAspectRatio="xMidYMid meet" aria-hidden="true" style={{opacity: faint ? .25 : ok ? .7 : 1}}>
      {/* vector-effect is not inherited, so each shape carries it: strokes and dots stay in screen pixels */}
      <g transform={`translate(0 ${dy})`} fill="none" stroke={ok ? "#3b82f6" : "rgba(255,255,255,.85)"} strokeWidth={ok ? 2.5 : 3}
        strokeDasharray={ok ? "none" : "0.1 9"} strokeLinecap="round" style={{transition:"stroke .3s"}}>
        <ellipse cx="80" cy="20" rx="6.5" ry="8.5" vectorEffect="non-scaling-stroke"/>
        <path d={`M 75 28 L 75 31 Q 66 31.5 62 35 Q 59 38 58.5 46 L 56 ${h - dy} M 85 28 L 85 31 Q 94 31.5 98 35 Q 101 38 101.5 46 L 104 ${h - dy}`} vectorEffect="non-scaling-stroke"/>
      </g>
    </svg>
  );
}

// Four rounded corners around the picture; their colour is the camera's state.
function FrameBox({state=""}){
  const corner = <svg viewBox="0 0 52 52" fill="none" aria-hidden="true"><path d="M4 48V28C4 14.745 14.745 4 28 4H48" stroke="currentColor" strokeWidth="4" strokeLinecap="round"/></svg>;
  return <div className={`frame-box ${state}`}>{corner}{corner}{corner}{corner}</div>;
}

// The camera view with its overlays. `children` go in the glass bar: over the
// bottom of the picture on wide screens, under it on phones. `pending` marks a
// result that waits for approval.
function Stage({live, model, sample, top, children, pending=false, offTitle="Kamera kapalı", offText}){
  const state = live.framing ? (live.framing.ok ? "ok" : "bad") : null;
  const off = !live.running && !sample;
  const frame = live.phase === "countdown" ? "countdown" : live.phase === "recording" ? "recording"
    : pending ? "pending" : state === "ok" ? "ok" : "";
  // The corners keep clear of the pills above and the glass bar below, whose height follows its contents
  const [barHeight, setBarHeight] = useState(0);
  const observer = useRef(null);
  const barRef = useCallback((el) => {
    observer.current?.disconnect();
    if (!el) return;
    observer.current = new ResizeObserver(() => setBarHeight(el.offsetHeight));
    observer.current.observe(el);
  }, []);
  const bars = `${live.running && top ? "has-top" : ""} ${live.running && children ? "has-bar" : ""}`;
  return (
    <div className={`stage stage-bg ${bars} ${live.phase === "recording" ? "recording" : ""} ${off ? "off" : ""}`} style={{"--bar-h":`${barHeight}px`}}>
      <div className="stage-media" style={{aspectRatio:String(live.aspect)}}>
        <video ref={live.videoRef} playsInline muted/>
        {(live.running || sample) && <Guide state={state} aspect={live.aspect} faint={live.phase === "recording"}/>}
        {(live.running || sample) && <FrameBox state={frame}/>}
        {live.running && top && <div className="stage-top">{top}</div>}
        {live.phase === "countdown" && <div className="count"><span key={live.count}>{live.count}</span></div>}
        {off && (
          <div className="stage-off">
            <div>
              <div className="badge glass"><Icon name="video-camera" size={34} duo/></div>
              <h3>{model === undefined ? "Bağlanıyor" : !model ? "Sunucuya ulaşılamıyor" : !model.ready ? "Model yükleniyor" : offTitle}</h3>
              <p>{model && !model.ready ? "İlk açılışta model birkaç saniyede belleğe yüklenir." : offText || "Kamerayı açınca kendinizi çizgiye yerleştirin; görüntü kaydedilmez."}</p>
              <button className="btn btn-primary lg" onClick={live.start} disabled={live.starting || !model?.ready}>
                {live.starting ? <span className="spinner"/> : <Icon name="video-camera" size={20}/>}{live.starting ? "Açılıyor" : "Kamerayı aç"}
              </button>
            </div>
          </div>
        )}
      </div>
      {live.running && children && <div className="stage-bar glass" ref={barRef}>{children}</div>}
    </div>
  );
}

// Record button: a 72 px ring that fills in the recording colour over the 4 seconds.
function RecordButton({live, onPress}){
  const r = 33, c = 2 * Math.PI * r;
  const busy = live.phase !== "idle";
  const filled = live.phase === "recording" ? live.progress : live.phase === "reading" ? 1 : 0;
  return (
    <button className={`rec ${busy ? "busy" : ""}`} onClick={onPress} disabled={!live.running || busy} aria-label={busy ? "Kaydediliyor" : "İşareti kaydet"}>
      <svg viewBox="0 0 72 72" aria-hidden="true">
        <circle cx="36" cy="36" r={r} fill="none" stroke="rgba(255,255,255,.2)" strokeWidth="3"/>
        <circle cx="36" cy="36" r={r} fill="none" stroke="#e11d48" strokeWidth="3.5" strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={c * (1 - filled)} transform="rotate(-90 36 36)"
          style={{transition: filled ? "stroke-dashoffset .1s linear" : "none", opacity: filled ? 1 : 0}}/>
      </svg>
      <span className="core"/>
    </button>
  );
}

function WatchMark({state}){
  return (
    <div className={`watch ${state}`} aria-hidden="true">
      <svg viewBox="0 0 72 72">
        <circle cx="36" cy="36" r="33" fill="none" stroke="rgba(255,255,255,.2)" strokeWidth="3"/>
        {state === "reading" && <circle cx="36" cy="36" r="33" fill="none" stroke="#3b82f6" strokeWidth="3.5" strokeLinecap="round" strokeDasharray="52 156"/>}
      </svg>
      <i/>
    </div>
  );
}

function CameraSelect({live, glass=false}){
  if (live.devices.length < 2) return null;
  return (
    <select className="select" value={live.deviceId} onChange={e => live.chooseDevice(e.target.value)} aria-label="Kamera"
      style={glass ? {height:38, background:"rgba(255,255,255,.1)", color:"var(--stage-ink)", borderColor:"var(--glass-line)", maxWidth:190} : {maxWidth:260}}>
      <option value="">Varsayılan kamera</option>
      {live.devices.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `Kamera ${i + 1}`}</option>)}
    </select>
  );
}
