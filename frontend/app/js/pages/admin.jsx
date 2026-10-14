// Admin: accounts, the model's measurements and the server log.

const ADMIN_TABS = [
  {value:"admin/overview", label:"Genel bakış", icon:"chart-bar"},
  {value:"admin/users", label:"Kullanıcılar", icon:"users"},
  {value:"admin/model", label:"Model", icon:"cpu"},
  {value:"admin/logs", label:"Kayıtlar", icon:"list-bullets"}
];

function AdminPage({tab, navigate, model}){
  return (
    <div className="page">
      <PageHead title="Yönetim" lead="Hesaplar, model ve sunucu kayıtları.">
        <Seg label="Yönetim bölümleri" value={tab} onChange={navigate} options={ADMIN_TABS}/>
      </PageHead>
      <div key={tab} className="rise">
        {tab === "admin/users" ? <AdminUsers/> : tab === "admin/model" ? <AdminModel/> : tab === "admin/logs" ? <AdminLogs/> : <AdminOverview model={model}/>}
      </div>
    </div>
  );
}

function AdminOverview({model}){
  const [ov, setOv] = useState(null);
  useEffect(() => { apiFetch(API.adminOverview).then(setOv).catch(() => setOv({})); }, []);
  const stats = [
    ["users", "Etkin hesap", ov?.active_users, `toplam ${ov?.total_users ?? "-"} hesap`],
    ["check-circle", "Onaylanan kelime", ov?.translations, "tüm kullanıcılar"],
    ["target", "Ortalama güven", ov?.avg_confidence ? trPercent(ov.avg_confidence) : "-", "son 100 kelimede"],
    ["cpu", "Model", model?.ready ? "Hazır" : (MODEL_STATES[model?.state] || "-"), model?.ready ? `${DEVICE_NAMES[model.device] || "-"} üzerinde` : (model?.message || "")]
  ];
  return (
    <div className="grid stagger" style={{gridTemplateColumns:"repeat(auto-fit,minmax(220px,1fr))"}}>
      {stats.map(([icon, label, value, sub]) => (
        <section className="panel stat" key={label}>
          <div className="top"><span className="stat-icon"><Icon name={icon} size={22} duo/></span>{label}</div>
          <div className="value num">{ov ? (value ?? "-") : <Skeleton height={44} width={80}/>}</div>
          <div className="sub">{sub}</div>
        </section>
      ))}
    </div>
  );
}

function AdminUsers(){
  const [rows, setRows] = useState(null);
  useEffect(() => { apiFetch(API.adminUsers).then(d => setRows(Array.isArray(d) ? d : [])).catch(() => setRows([])); }, []);
  if (!rows) return <div className="panel panel-pad"><Skeleton height={240}/></div>;
  return (
    <section className="panel table-wrap">
      <table className="table">
        <thead><tr><th>Kişi</th><th>Rol</th><th>Durum</th><th style={{textAlign:"right"}}>Oturum</th></tr></thead>
        <tbody>{rows.map((u, i) => (
          <tr key={u.id || i}>
            <td><div className="who"><span className="initials">{initials(u)}</span><div><strong>{u.name || u.full_name || "-"}</strong><span>{u.email}</span></div></div></td>
            <td>{u.role === "Admin" ? <span className="chip brand"><Icon name="shield-check" size={15}/>Yönetici</span> : <span className="chip">Kullanıcı</span>}</td>
            <td><StatusDot tone={u.status === "Active" ? "good" : "warn"}>{u.status === "Active" ? "Etkin" : (u.status || "-")}</StatusDot></td>
            <td className="num" style={{textAlign:"right", fontWeight:700}}>{u.sessions ?? "-"}</td>
          </tr>
        ))}</tbody>
      </table>
    </section>
  );
}

function AdminModel(){
  const [m, setM] = useState(null);
  useEffect(() => { apiFetch(API.adminModel).then(setM).catch(() => setM({})); }, []);
  if (!m) return <div className="panel panel-pad"><Skeleton height={260}/></div>;
  const model = m.model_226 || {};
  const frames = model.record_frames || RECORD_FRAMES;
  const timings = [["Kare", model.avg_frame_ms, "kişi tespiti ve poz"], ["Kişi tespiti", model.avg_detect_ms, `${model.detect_every ?? 1} karede bir`], ["Poz", model.avg_pose_ms, "RTMW"]];
  const slowest = Math.max(60, ...timings.map(t => Number(t[1]) || 0));
  const streams = Object.entries(m.streams || {});
  const kv = (k, v, s) => <div className="kv" key={k}><div className="k">{k}</div><div className="v">{String(v)}</div>{s && <div className="s">{s}</div>}</div>;
  return (
    <div className="kv-grid">
      <section className="panel">
        <div className="panel-head"><span className="panel-icon"><Icon name="cpu" size={20} duo/></span><div><h3>Durum</h3></div></div>
        <div style={{marginTop:8}}>
          {kv("Model", MODEL_STATES[model.state] || "-", model.message || `${model.num_classes ?? 226} işaret`)}
          {kv("Cihaz", DEVICE_NAMES[model.device] || "-", "RTMW ve YOLOX, ONNX Runtime")}
          {kv("Düğmesiz çeviri", model.continuous ? "Açık" : "Kapalı", "kare başına en çok 60 ms")}
        </div>
      </section>
      <section className="panel">
        <div className="panel-head"><span className="panel-icon"><Icon name="timer" size={20} duo/></span><div><h3>Süreler</h3><p>{model.avg_frame_ms != null ? "Kare başına ortalama" : "Bu oturumda henüz kare işlenmedi"}</p></div></div>
        <div style={{marginTop:12, paddingBottom:12}}>
          {timings.map(([k, v, s], i) => (
            <div className="timing" key={k} title={s}>
              <span>{k}</span>
              <Meter value={v != null ? Number(v) / slowest : 0} kind="brand" delay={i * 90}/>
              <b className={v != null ? "" : "muted"}>{v != null ? `${v} ms` : "yok"}</b>
            </div>
          ))}
        </div>
      </section>
      <section className="panel">
        <div className="panel-head"><span className="panel-icon"><Icon name="record" size={20} duo/></span><div><h3>Kayıt ve karar</h3></div></div>
        <div style={{marginTop:8}}>
          {kv("Kayıt", `${Math.round(frames / (model.camera_fps || 12))} sn`, `${frames} kare; kesilen işaretten ${model.frame_count ?? 32} kare`)}
          {kv("Onay eşiği", percent(m.thresholds?.top1 ?? .7), `ikinciden en az ${percent(m.thresholds?.margin ?? .12)} önde`)}
          {kv("Avatar", m.landmark_words ?? "-", "animasyonu olan işaret")}
        </div>
      </section>
      <section className="panel">
        <div className="panel-head"><span className="panel-icon"><Icon name="stack" size={20} duo/></span><div><h3>Akışlar</h3><p>Topluluktaki ağırlıklar</p></div></div>
        <div style={{marginTop:12, paddingBottom:12}}>
          {streams.length ? streams.map(([name, weight], i) => (
            <div className="timing" key={name}><span style={{overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap"}} title={name}>{name}</span><Meter value={Number(weight)} delay={i * 90}/><b>{percent(weight)}</b></div>
          )) : <div className="kv"><div className="s">Bilgi yok</div></div>}
        </div>
      </section>
    </div>
  );
}

const LOG_LEVELS = {Success:["check-circle", "Başarılı"], Warning:["warning", "Uyarı"], Error:["x-circle", "Hata"], Info:["info", "Bilgi"]};

function AdminLogs(){
  const [rows, setRows] = useState(null);
  const [level, setLevel] = useState("all");
  useEffect(() => { apiFetch(API.adminLogs).then(d => setRows(Array.isArray(d) ? d : [])).catch(() => setRows([])); }, []);
  const shown = (rows || []).filter(l => level === "all" || l.level === level);
  return (
    <>
      <div style={{marginBottom:16}}>
        <Seg label="Seviye" value={level} onChange={setLevel}
          options={[{value:"all", label:"Tümü"}, {value:"Success", label:"Başarılı"}, {value:"Info", label:"Bilgi"}, {value:"Warning", label:"Uyarı"}, {value:"Error", label:"Hata"}]}/>
      </div>
      {!rows ? <div className="panel panel-pad"><Skeleton height={260}/></div> : (
        <section className="panel table-wrap">
          {shown.length ? (
            <table className="table">
              <thead><tr><th>Zaman</th><th>Seviye</th><th>Kayıt</th><th>Kullanıcı</th></tr></thead>
              <tbody>{shown.map((l, i) => {
                const [icon, name] = LOG_LEVELS[l.level] || ["info", l.level];
                return (
                  <tr key={l.id || i}>
                    <td className="muted num" style={{whiteSpace:"nowrap"}}>{formatTime(l.timestamp, true)}</td>
                    <td><span className={`log-level ${LOG_LEVELS[l.level] ? l.level : "Info"}`}><Icon name={icon} size={18} duo/>{name}</span></td>
                    <td>{l.message}</td>
                    <td className="muted">{l.user || "-"}</td>
                  </tr>
                );
              })}</tbody>
            </table>
          ) : <Empty icon="list-bullets" title="Kayıt yok">Bu seviyede kayıt bulunmuyor.</Empty>}
        </section>
      )}
    </>
  );
}
