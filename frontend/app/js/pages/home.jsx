// Home: where to start, the sign of the day, and what you have approved so far.

function HomePage({user, model, navigate}){
  const [rows, setRows] = useState(null);
  const [summary, setSummary] = useState(null);
  const [daily, setDaily] = useState(null);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    apiFetch(API.history).then(d => setRows(Array.isArray(d) ? d : [])).catch(() => setRows([]));
    apiFetch(API.summary).then(setSummary).catch(() => setSummary({translations:0, avg_confidence:0}));
    apiFetch(API.dictionary).then(d => {
      const signs = (Array.isArray(d) ? d : []).filter(r => r.animation_key);
      if (signs.length) setDaily(signs[Math.floor(Date.now() / 86400000) % signs.length]);
    }).catch(() => {});
  }, []);

  const name = firstName(user);
  const today = new Date().toLocaleDateString("tr-TR", {weekday:"long", day:"numeric", month:"long"});
  const week = rows ? dailyCounts(rows, 7) : null;
  const weekMax = week ? Math.max(1, ...week.map(d => d.count)) : 1;
  const weekTotal = week ? week.reduce((sum, d) => sum + d.count, 0) : 0;
  const avg = summary?.translations ? Number(summary.avg_confidence) / 100 : 0;
  const recent = (rows || []).slice(0, 12);
  const dailySign = daily ? {gloss:daily.animation_key, text:daily.text} : null;

  return (
    <div className="page">
      <PageHead eyebrow={today} title={name ? `Merhaba, ${name}` : "Merhaba"} lead="Bugün hangi işaretleri çevirelim?"/>

      <div className="home-top">
        <section className="hero stage-bg on-stage rise">
          <BrandArt/>
          <div className="hero-status"><ModelStatus model={model}/></div>
          <h2>Bir işaret yapın, <em>kelimeye</em> dönüşsün.</h2>
          <p>Kamera dört saniye kaydeder ve SignTurk işareti tanır. Siz onaylayınca kelime cümleye eklenir.</p>
          <div className="actions">
            <button className="btn btn-primary lg" onClick={() => navigate("live")}><Icon name="video-camera" size={20}/>Çeviriye başla</button>
            <button className="btn btn-glass lg" onClick={() => navigate("dictionary")}><Icon name="book-open-text" size={20}/>Sözlüğe göz at</button>
          </div>
        </section>

        <section className="panel daily-card rise" style={{animationDelay:"60ms"}}>
          <div className="daily-stage stage-bg">
            {playing && dailySign ? <AvatarFrame sign={dailySign}/> : (
              <>
                <img src={AVATAR_POSES.rest} alt="" onError={e => { e.currentTarget.style.display = "none"; }}/>
                <button className="daily-play" onClick={() => setPlaying(true)} disabled={!daily} aria-label={daily ? `"${daily.text}" işaretini oynat` : "Oynat"}>
                  <Icon name="play" size={26}/>
                </button>
              </>
            )}
          </div>
          <div className="daily-body">
            <span className="eyebrow"><Icon name="calendar" size={16} duo/>Günün işareti</span>
            {daily ? <div className="word">{daily.text}</div> : <Skeleton height={36} width={150} style={{marginTop:4}}/>}
            <div className="muted">{daily?.en ? `İngilizcesi: ${daily.en}` : " "}</div>
            <div className="row">
              <button className="btn btn-secondary sm" disabled={!daily} onClick={() => setPlaying(true)}><Icon name="play-circle" size={18} duo/>{playing ? "Oynatılıyor" : "Oynat"}</button>
              <button className="btn btn-ghost sm" onClick={() => navigate("dictionary")}>Sözlükte gör<Icon name="arrow-right" size={16}/></button>
            </div>
          </div>
        </section>
      </div>

      <div className="grid cols-3 stagger" style={{marginTop:20}}>
        <section className="panel stat">
          <div className="top"><span className="stat-icon"><Icon name="check-circle" size={22} duo/></span>Onaylanan kelime</div>
          <div className="value num">{summary ? summary.translations : <Skeleton height={44} width={90}/>}</div>
          <div className="sub">{week ? `Son 7 günde ${weekTotal}` : " "}</div>
          <div className="bars" aria-hidden="true">
            {(week || Array.from({length:7}, () => ({count:0}))).map((d, i) => (
              <i key={i} className={d.today && d.count ? "today" : d.count ? "has" : ""} style={{height:`${Math.max(8, d.count / weekMax * 100)}%`, animationDelay:`${i * 40}ms`}}/>
            ))}
          </div>
        </section>
        <section className="panel stat">
          <div className="top"><span className="stat-icon"><Icon name="target" size={22} duo/></span>Ortalama güven</div>
          <div className="value num">{summary ? (summary.translations ? trPercent(summary.avg_confidence) : "-") : <Skeleton height={44} width={110}/>}</div>
          <div className="sub">Onayladığınız kelimelerde</div>
          <div style={{marginTop:22}}><Meter value={avg}/></div>
        </section>
        <section className="panel stat">
          <div className="top"><span className="stat-icon"><Icon name="cpu" size={22} duo/></span>Model</div>
          <div className="value" style={{fontSize:34, marginTop:4}}>{model?.ready ? "Hazır" : model ? (MODEL_STATES[model.state] || "Hazır değil") : model === null ? "Bağlantı yok" : "…"}</div>
          <div className="sub">{model?.ready ? `${DEVICE_NAMES[model.device] || "-"} üzerinde, ${model.num_classes || 226} işaret` : (model?.message || "Sunucuya bağlanılıyor")}</div>
          <div style={{marginTop:16}}>
            {model?.ready && <span className={`chip ${model.continuous ? "good" : ""}`}><Icon name="hand-waving" size={16}/>{model.continuous ? "Düğmesiz çeviri açık" : "Düğmesiz çeviri GPU ister"}</span>}
          </div>
        </section>
      </div>

      <div className="grid cols-2" style={{marginTop:20}}>
        <section className="panel rise">
          <div className="panel-head"><span className="panel-icon"><Icon name="lightbulb" size={20} duo/></span><div><h3>Nasıl çalışır</h3><p>Üç adımda bir kelime</p></div></div>
          <div className="steps">
            <div className="step-card"><Icon name="frame-corners" size={30} duo/><span className="n">01</span><strong>Çizgiye yerleşin</strong><span>Belinizden yukarısı görünsün; köşeler maviye döner.</span></div>
            <div className="step-card"><Icon name="hand-waving" size={30} duo/><span className="n">02</span><strong>İşareti yapın</strong><span>Bitince ellerinizi indirin; işaret oradan kesilir.</span></div>
            <div className="step-card"><Icon name="check-circle" size={30} duo/><span className="n">03</span><strong>Onaylayın</strong><span>Doğru dediğiniz kelimeler cümleye eklenir.</span></div>
          </div>
        </section>
        <section className="panel rise">
          <div className="panel-head"><span className="panel-icon"><Icon name="clock-counter-clockwise" size={20} duo/></span><div><h3>Son kelimeler</h3><p>En son onayladıklarınız</p></div>
            <span className="spacer"/>{recent.length > 0 && <button className="btn btn-ghost sm" onClick={() => navigate("history")}>Tümü<Icon name="arrow-right" size={16}/></button>}</div>
          <div className="panel-pad">
            {rows === null ? <Skeleton height={38}/> : recent.length ? (
              <div className="tokens stagger">{recent.map((r, i) => <span className="token" key={r.id ?? i}>{r.text || r.result}</span>)}</div>
            ) : <Empty icon="chat-circle-text" title="Henüz kelime yok" action={<button className="btn btn-primary" onClick={() => navigate("live")}><Icon name="video-camera" size={18}/>İlk işareti yapın</button>}>
                Çeviri sayfasında bir işareti onaylayınca burada görünür.</Empty>}
          </div>
        </section>
      </div>
    </div>
  );
}
