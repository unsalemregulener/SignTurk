// History: the last two weeks at a glance, then every approved word by day.

function HistoryPage({navigate}){
  const [rows, setRows] = useState(null);
  useEffect(() => {
    apiFetch(API.history).then(d => setRows(Array.isArray(d) ? d : [])).catch(() => setRows([]));
  }, []);

  const days = rows ? dailyCounts(rows, 14) : null;
  const max = days ? Math.max(1, ...days.map(d => d.count)) : 1;
  const fortnight = days ? days.reduce((sum, d) => sum + d.count, 0) : 0;
  const values = (rows || []).map(r => confidenceOf(r.confidence)).filter(v => v != null);
  const average = values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  const groups = [];
  for (const row of rows || []){
    const label = row.created_at ? dayLabel(row.created_at) : "Tarihsiz";
    if (!groups.length || groups[groups.length - 1].label !== label) groups.push({label, rows:[]});
    groups[groups.length - 1].rows.push(row);
  }

  return (
    <div className="page">
      <PageHead title="Geçmiş" lead="Onayladığınız kelimeler; en yenisi en üstte."/>

      <section className="panel chart rise">
        <div className="chart-top">
          <div><div className="k">Son 14 gün</div><div className="v num">{days ? fortnight : "-"}<span className="muted" style={{fontSize:16, fontWeight:600, marginLeft:8}}>kelime</span></div></div>
          <div><div className="k">Ortalama güven</div><div className="v num">{average != null ? percent(average) : "-"}</div></div>
          <div><div className="k">Kayıtlı kelime</div><div className="v num">{rows ? rows.length : "-"}</div></div>
        </div>
        <div className="chart-bars" role="img" aria-label={`Son 14 günde ${fortnight} kelime`}>
          {(days || Array.from({length:14}, () => ({count:0, date:new Date()}))).map((d, i) => (
            <div className="col" key={i}>
              <div className={`bar ${d.count ? "has" : ""} ${d.today && d.count ? "today" : ""}`} style={{height:`${Math.max(3, d.count / max * 100)}%`, animationDelay:`${i * 35}ms`}}>
                {d.count > 0 && <b className="num">{d.count}</b>}
              </div>
              <span className="day">{d.today ? "Bugün" : d.date.toLocaleDateString("tr-TR", {day:"numeric"})}</span>
            </div>
          ))}
        </div>
      </section>

      {rows === null ? (
        <div className="day-group"><Skeleton height={26} width={140} style={{marginBottom:12}}/><Skeleton height={220}/></div>
      ) : !rows.length ? (
        <div className="panel" style={{marginTop:24}}>
          <Empty image={AVATAR_POSES.yes} title="Henüz kelime yok"
            action={<button className="btn btn-primary" onClick={() => navigate("live")}><Icon name="video-camera" size={18}/>Çeviriye başla</button>}>
            Çeviri sayfasında bir işareti onaylayınca burada görünür.
          </Empty>
        </div>
      ) : groups.map((g, gi) => (
        <section className="day-group rise" key={g.label} style={{animationDelay:`${Math.min(gi, 4) * 60}ms`}}>
          <div className="day-head"><h2>{g.label}</h2><span>{g.rows.length} kelime</span></div>
          <div className="panel entries">
            {g.rows.map((r, i) => {
              const value = confidenceOf(r.confidence);
              return (
                <div className="entry" key={r.id ?? i}>
                  <time dateTime={r.created_at}>{formatTime(r.created_at)}</time>
                  <strong>{r.text || r.result}</strong>
                  {value != null ? <Meter value={value} thin/> : <span/>}
                  <em>{trPercent(r.confidence)}</em>
                </div>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}
