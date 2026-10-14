// Dictionary: every sign the model knows, grouped by letter, with the avatar beside.

const TR_ALPHABET = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ".split("");

function DictionaryPage({navigate}){
  const [rows, setRows] = useState(null);
  const [search, setSearch] = useState("");
  const [letter, setLetter] = useState("");
  const [selected, setSelected] = useState(null);
  const [narrow, setNarrow] = useState(() => window.matchMedia("(max-width:1180px)").matches);
  useEffect(() => {
    apiFetch(API.dictionary).then(d => setRows(Array.isArray(d) ? d : [])).catch(() => setRows([]));
    const query = window.matchMedia("(max-width:1180px)");
    const onChange = () => setNarrow(query.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  const needle = search.trim().toLocaleLowerCase("tr-TR");
  const sorted = useMemo(() => (rows || []).slice().sort((a, b) => a.text.localeCompare(b.text, "tr")), [rows]);
  const firstLetters = useMemo(() => new Set(sorted.map(r => upperTR(r.text)[0])), [sorted]);
  const filtered = sorted.filter(r => (!needle || `${r.text} ${r.tr} ${r.en || ""} ${r.class_id}`.toLocaleLowerCase("tr-TR").includes(needle))
    && (!letter || upperTR(r.text)[0] === letter));
  const groups = [];
  for (const r of filtered){
    const key = upperTR(r.text)[0];
    if (!groups.length || groups[groups.length - 1].key !== key) groups.push({key, rows:[]});
    groups[groups.length - 1].rows.push(r);
  }
  const pick = (r) => setSelected({gloss:r.animation_key || r.tr, text:r.text, en:r.en, id:r.class_id, avatar:!!r.animation_key});

  return (
    <div className="page">
      <PageHead title="Sözlük" lead={`SignTurk'ün tanıdığı ${rows?.length || 226} işaret. Birine dokunun, avatar nasıl yapıldığını göstersin.`}>
        <div className="input-wrap search-big">
          <Icon name="magnifying-glass" size={20}/>
          <input className="input" placeholder="Türkçe ya da İngilizce ara" value={search} onChange={e => setSearch(e.target.value)} aria-label="İşaret ara"/>
          {search && <span className="end"><button className="btn btn-ghost sm icon-only" onClick={() => setSearch("")} aria-label="Aramayı temizle"><Icon name="x" size={18}/></button></span>}
        </div>
      </PageHead>

      <div className="alpha rise" role="group" aria-label="Harfe göre süz">
        <button aria-pressed={!letter} onClick={() => setLetter("")} style={{padding:"0 14px"}}>Tümü</button>
        {TR_ALPHABET.map(l => <button key={l} aria-pressed={letter === l} disabled={!firstLetters.has(l)} onClick={() => setLetter(letter === l ? "" : l)}>{l}</button>)}
      </div>

      <div className="dict">
        <div>
          {rows === null ? (
            <div className="signs">{Array.from({length:18}, (_, i) => <Skeleton key={i} height={66}/>)}</div>
          ) : filtered.length ? groups.map(g => (
            <section key={g.key}>
              <div className="letter"><h2>{g.key}</h2><span>{g.rows.length} işaret</span></div>
              <div className="signs stagger">
                {g.rows.map(r => (
                  <button key={r.class_id} className="sign" aria-pressed={selected?.id === r.class_id} onClick={() => pick(r)}>
                    <strong>{r.text}</strong><span>{r.en || "-"}</span>
                    {r.animation_key && <Icon name="play-circle" size={18} duo={selected?.id === r.class_id}/>}
                  </button>
                ))}
              </div>
            </section>
          )) : <div className="panel"><Empty icon="magnifying-glass" title="Bulunamadı">"{search}" ile eşleşen bir işaret yok. Başka bir yazım deneyin.</Empty></div>}
        </div>

        {!narrow && (
          <aside className="panel raised dict-side">
            {selected ? (
              <>
                {selected.avatar
                  ? <AvatarFrame sign={selected}/>
                  : <div className="avatar-frame stage-bg" style={{height:430, display:"grid", placeItems:"center", color:"var(--stage-ink)"}}>
                      <div style={{textAlign:"center"}}><Icon name="hand-palm" size={40} duo/><p style={{marginTop:10, opacity:.75}}>Bu işaretin animasyonu yok.</p></div></div>}
                <div className="info">
                  <div className="eyebrow">AUTSL · {selected.id}</div>
                  <h2>{selected.text}</h2>
                  <div className="muted">İngilizcesi: {selected.en || "-"}</div>
                  <div className="row">
                    <button className="btn btn-primary" onClick={() => navigate("live")}><Icon name="video-camera" size={18}/>Çeviride dene</button>
                  </div>
                </div>
              </>
            ) : (
              <div className="dict-empty stage-bg on-stage">
                <BrandArt/>
                <img src={AVATAR_POSES.rest} alt="" onError={e => { e.currentTarget.style.display = "none"; }}/>
                <div className="dict-empty-copy">
                  <div className="eyebrow"><Icon name="hand-waving" size={16} duo/>Avatar</div>
                  <h3>Bir işaret seçin, nasıl yapıldığını göstereyim.</h3>
                </div>
              </div>
            )}
          </aside>
        )}
      </div>
      {narrow && selected && <HowToDrawer sign={selected} onClose={() => setSelected(null)}/>}
    </div>
  );
}
