// Settings: appearance and camera live in this browser; voice and avatar
// speed are saved to the account as soon as they change.

function SettingsPage({user, navigate, onLogout}){
  const [settings, setSettings] = useState(null);
  const [theme, setTheme] = useTheme();
  const [toast, showToast] = useToast();
  const [cameras, setCameras] = useState([]);
  const [camera, setCamera] = useState(() => readKey(KEYS.camera) || "");
  const saveTimer = useRef(null);
  useEffect(() => {
    apiFetch(API.settings).then(d => setSettings({tts_enabled:false, speech_rate:1, avatar_speed:1, ...d})).catch(() => setSettings({tts_enabled:false, speech_rate:1, avatar_speed:1}));
    navigator.mediaDevices?.enumerateDevices?.().then(list => setCameras(list.filter(d => d.kind === "videoinput" && d.deviceId))).catch(() => {});
    return () => clearTimeout(saveTimer.current);
  }, []);

  const change = (patch) => {
    const next = {...settings, ...patch};
    setSettings(next);
    clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      apiFetch(API.settings, {method:"POST", body:JSON.stringify(next)})
        .then(() => showToast("Kaydedildi"))
        .catch(err => showToast(err.message || "Kaydedilemedi", "bad"));
    }, 450);
  };
  const reopenSetup = () => { writeKey(KEYS.setup, null); navigate("live"); };
  const chooseCamera = (id) => { setCamera(id); writeKey(KEYS.camera, id || null); showToast("Kamera seçildi"); };
  const s = settings || {tts_enabled:false, speech_rate:1, avatar_speed:1};

  return (
    <div className="page">
      <PageHead title="Ayarlar" lead="Görünüm, ses, avatar ve kamera tercihleri."/>
      <div className="settings">
        <section className="panel wide rise">
          <div className="panel-head"><span className="panel-icon"><Icon name="palette" size={20} duo/></span><div><h3>Görünüm</h3><p>Sistem seçiliyse bilgisayarınızın ya da telefonunuzun açık/koyu ayarını izler.</p></div></div>
          <div className="themes" role="group" aria-label="Tema">
            {[["system", "Sistem", "monitor"], ["light", "Açık", "sun"], ["dark", "Koyu", "moon"]].map(([value, label, icon]) => (
              <button key={value} className="theme-opt" aria-pressed={theme === value} onClick={() => setTheme(value)}>
                <span className={`mini ${value}`} aria-hidden="true">
                  <span className="r"><i/><i/><i/><i/></span>
                  <span className="m"><i className="s"/><i className="t"/><i className="w"/></span>
                </span>
                <span style={{display:"flex", gap:8, alignItems:"center"}}><Icon name={icon} size={18} duo={theme === value}/>{label}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="panel rise">
          <div className="panel-head"><span className="panel-icon"><Icon name="speaker-high" size={20} duo/></span><div><h3>Ses</h3><p>Tarayıcının Türkçe sesiyle okunur.</p></div></div>
          <div style={{marginTop:10}}>
            <div className="setting">
              <div className="text"><strong>Onaylanan kelimeyi oku</strong><span>Her onaydan sonra kelime sesli okunur</span></div>
              <Switch checked={s.tts_enabled} onChange={v => change({tts_enabled:v})} label="Onaylanan kelimeyi oku"/>
            </div>
            <div className="setting col">
              <div style={{display:"flex", alignItems:"center", gap:12}}>
                <div className="text"><strong>Konuşma hızı</strong><span>Seslendir düğmesi de bu hızı kullanır</span></div>
                <span className="val">{Number(s.speech_rate).toLocaleString("tr-TR", {minimumFractionDigits:1, maximumFractionDigits:1})}×</span>
              </div>
              <div style={{display:"flex", gap:14, alignItems:"center"}}>
                <Range value={Number(s.speech_rate)} min={0.5} max={2} step={0.1} onChange={v => change({speech_rate:v})} label="Konuşma hızı"/>
                <button className="btn btn-secondary sm" onClick={() => speak("Merhaba, ben SignTurk.", Number(s.speech_rate))}><Icon name="play" size={16}/>Dinle</button>
              </div>
            </div>
          </div>
        </section>

        <section className="panel rise">
          <div className="panel-head"><span className="panel-icon"><Icon name="hand-waving" size={20} duo/></span><div><h3>Avatar</h3><p>Sözlük ve "Nasıl yapılır?" animasyonları.</p></div></div>
          <div style={{marginTop:10}}>
            <div className="setting col">
              <div style={{display:"flex", alignItems:"center", gap:12}}>
                <div className="text"><strong>Oynatma hızı</strong><span>Yavaş oynatma işareti izlemeyi kolaylaştırır</span></div>
                <span className="val">{Number(s.avatar_speed).toLocaleString("tr-TR", {minimumFractionDigits:2, maximumFractionDigits:2})}×</span>
              </div>
              <Range value={Number(s.avatar_speed)} min={0.25} max={3} step={0.05} onChange={v => change({avatar_speed:v})} label="Avatar hızı"/>
            </div>
          </div>
        </section>

        <section className="panel rise">
          <div className="panel-head"><span className="panel-icon"><Icon name="video-camera" size={20} duo/></span><div><h3>Kamera</h3><p>Bu tarayıcıda hatırlanır.</p></div></div>
          <div style={{marginTop:10}}>
            <div className="setting">
              <div className="text"><strong>Kamera kurulumu</strong><span>Kadraj ayarını ve deneme işaretini yeniden açar</span></div>
              <button className="btn btn-secondary sm" onClick={reopenSetup}><Icon name="frame-corners" size={16}/>Yeniden aç</button>
            </div>
            <div className="setting">
              <div className="text"><strong>Varsayılan kamera</strong><span>{cameras.length ? "Çeviri bu kamerayla açılır" : "Kamera izni verildikten sonra listelenir"}</span></div>
              <select className="select" value={camera} onChange={e => chooseCamera(e.target.value)} disabled={!cameras.length} style={{maxWidth:220}} aria-label="Varsayılan kamera">
                <option value="">Varsayılan</option>
                {cameras.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `Kamera ${i + 1}`}</option>)}
              </select>
            </div>
          </div>
        </section>

        <section className="panel rise">
          <div className="panel-head"><span className="panel-icon"><Icon name="user-circle" size={20} duo/></span><div><h3>Hesap</h3><p>Oturum bu tarayıcıda açık.</p></div></div>
          <div style={{marginTop:10}}>
            <div className="setting">
              <span className="initials" style={{width:46, height:46, fontSize:16}}>{initials(user)}</span>
              <div className="text"><strong>{user?.full_name || "Kullanıcı"}</strong><span>{user?.email}</span></div>
              <button className="btn btn-danger sm" onClick={onLogout}><Icon name="sign-out" size={16}/>Çıkış yap</button>
            </div>
          </div>
        </section>
      </div>
      {toast}
    </div>
  );
}
