// Translate: the camera, the recognised sign, and the sentence built from approved signs.
// On the first visit a three-step camera setup covers the page.

const TEST_SIGNS = [{gloss:"evet", text:"evet"}, {gloss:"hayir", text:"hayır"}, {gloss:"tesekkur", text:"teşekkür"}];

function TranslatePage({model, onExpired}){
  const live = useLive(model, onExpired);
  const [setupOpen, setSetupOpen] = useState(() => readKey(KEYS.setup) !== "1");
  const [result, setResult] = useState(null);
  const [resultKey, setResultKey] = useState(0);
  const [pending, setPending] = useState(null);
  const [added, setAdded] = useState(null);
  const [words, setWords] = useState([]);
  const [sentence, setSentence] = useState("");
  const [how, setHow] = useState(null);
  const [freshKey, setFreshKey] = useState(0);
  const speech = useRef({on:false, rate:1});
  const keys = useRef({});
  const pointer = useRef(false);

  useEffect(() => {
    apiFetch(API.settings).then(s => { speech.current = {on:!!s?.tts_enabled, rate:Number(s?.speech_rate) || 1}; }).catch(() => {});
  }, []);

  if (!setupOpen) live.listener.current = (data) => {
    if (data.type === "result"){
      setAdded(null);
      if (data.error){ setResult({error:data.error}); setPending(null); return; }
      setResult(data); setResultKey(k => k + 1);
      setPending(data.pending_word || null);
    } else if (data.type === "sentence_update"){
      const next = Array.isArray(data.display_words) ? data.display_words : (data.words || []);
      setWords(next); setSentence(data.sentence || ""); setPending(null);
      if (data.added){
        const word = next[next.length - 1] || data.added;
        setAdded(word); setFreshKey(k => k + 1);
        if (speech.current.on) speak(word, speech.current.rate);
      }
    } else if (data.type === "rejected"){ setPending(null); setResult(null); }
  };

  const finishSetup = () => { writeKey(KEYS.setup, "1"); setSetupOpen(false); };
  const record = () => { if (live.record()){ setResult(null); setAdded(null); setPending(null); live.setError(""); } };
  const approve = (gloss) => { if (live.send(gloss ? {action:"approve", word:gloss} : {action:"approve"})) setPending(null); };
  const reject = () => { live.send({action:"reject"}); setPending(null); setResult(null); };
  const undo = () => { live.send({action:"undo"}); setAdded(null); };
  const chooseMode = (next) => { live.setMode(next); setResult(null); setPending(null); setAdded(null); };

  // Space records, Enter approves, Esc rejects. A button focused from the
  // keyboard keeps Space and Enter; one focused by a click does not.
  keys.current = {record, approve, reject, pending, running:live.running, mode:live.mode, blocked:!!how || setupOpen};
  useEffect(() => {
    const onPointer = () => { pointer.current = true; };
    const onKey = (e) => {
      if (e.key === "Tab"){ pointer.current = false; return; }
      const k = keys.current, el = e.target;
      if (k.blocked || /^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName)) return;
      const control = /^(BUTTON|A)$/.test(el.tagName);
      if (control && !pointer.current && (e.code === "Space" || e.key === "Enter")) return;
      if (e.code === "Space"){ if (k.running && k.mode === "record"){ e.preventDefault(); el.blur?.(); k.record(); } }
      else if (e.key === "Enter" && k.pending){ e.preventDefault(); k.approve(); }
      else if (e.key === "Escape" && k.pending) k.reject();
    };
    window.addEventListener("pointerdown", onPointer, true);
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("pointerdown", onPointer, true); window.removeEventListener("keydown", onKey); };
  }, []);

  const line = framingLine(live);
  const continuous = live.mode === "continuous";
  const top = result?.top || [];
  const best = top[0];
  const sure = !!result?.confident;
  const reading = live.phase === "reading" || live.watch === "reading";

  let view = "idle";
  if (live.phase === "countdown") view = "countdown";
  else if (live.phase === "recording") view = "recording";
  else if (reading) view = "reading";
  else if (result?.error) view = "error";
  else if (added) view = "added";
  else if (best) view = sure ? "sure" : "unsure";

  const sub = continuous
    ? ({signing:"İşaret yapılıyor", reading:"Okunuyor"}[live.watch] || "İşaret yapın, sonra ellerinizi indirin")
    : ({countdown:"Hazırlanın", recording:"Kaydediliyor; bitince ellerinizi indirin", reading:"Okunuyor"}[live.phase]
       || (TOUCH ? "Kaydetmek için düğmeye basın" : "Kaydetmek için düğmeye ya da Boşluk'a basın"));

  const recSecs = (live.progress * 4).toLocaleString("tr-TR", {minimumFractionDigits:1, maximumFractionDigits:1});
  const topPills = (
    <>
      <span className={`pill glass ${live.phase === "recording" ? "rec-on" : ""}`}>
        <span className={`dot ${live.phase === "recording" || live.watch === "signing" ? "live" : "brand"}`}/>
        {live.phase === "recording" ? `Kayıt · ${recSecs} sn` : continuous ? "Düğmesiz çeviri" : "Kamera açık"}
      </span>
      <span className="spacer"/>
      {continuous && live.auto && <span className="pill glass"><Icon name="lightning" size={16} duo/>Otomatik onay</span>}
    </>
  );
  const awaiting = view === "sure" && !!pending;
  const showHow = (view === "sure" || view === "unsure" || view === "added") && (best || added);

  // While the setup is open it is the only camera view, so the one <video> keeps the stream.
  if (setupOpen) return <Setup live={live} model={model} onDone={finishSetup}/>;

  return (
    <div className="page">
      <PageHead title="Çeviri" lead="Kameraya bir işaret yapın; SignTurk tanısın, siz onaylayın.">
        <Seg label="Çeviri biçimi" value={live.mode} onChange={chooseMode} disabled={live.phase !== "idle"}
          options={[{value:"record", label:"Kayıtla", icon:"record"},
                    {value:"continuous", label:"Düğmesiz", icon:"hand-waving", disabled:model ? !model.continuous : true,
                     title:model?.continuous ? "Her işaret kendiliğinden okunur" : "Düğmesiz çeviri için sunucunun GPU ile çalışması gerekir"}]}/>
      </PageHead>

      <div className="translate">
        <div className="rise">
          <Stage live={live} model={model} top={topPills} pending={awaiting}>
            {continuous ? <WatchMark state={live.watch}/> : <RecordButton live={live} onPress={record}/>}
            <div className="text">
              <strong className={line.tone}><Icon name={line.tone === "good" ? "check-circle" : line.tone === "bad" ? "warning" : "scan"} size={18}/>{line.text}</strong>
              <span>{sub}</span>
            </div>
            <div className="side">
              {continuous && (
                <label style={{display:"flex", alignItems:"center", gap:10, fontSize:14, fontWeight:600}} title="Açıkken emin olunan işaret onay beklemeden cümleye girer">
                  Otomatik onay<Switch checked={live.auto} onChange={live.setAuto} label="Otomatik onay" glass/>
                </label>
              )}
              {!continuous && !TOUCH && live.phase === "idle" && <kbd>Boşluk</kbd>}
              <CameraSelect live={live} glass/>
              <button className="btn btn-glass icon-only" onClick={live.stop} aria-label="Kamerayı kapat" title="Kamerayı kapat"><Icon name="video-camera-slash" size={20}/></button>
            </div>
          </Stage>
          {live.error && <div className="note bad" style={{marginTop:14}} role="alert"><Icon name="warning-circle" size={20}/>{live.error}</div>}
        </div>

        <div className="dock rise" style={{animationDelay:"60ms"}}>
          <section className={`panel approval ${awaiting ? "pending" : ""} ${view === "added" ? "locked" : ""}`} aria-live="polite">
            <div className="approval-head">
              {view === "added" ? <span className="eyebrow" style={{color:"var(--good)"}}><Icon name="check-circle" size={16} duo/>Cümleye eklendi</span>
                : view === "unsure" ? <span className="eyebrow"><Icon name="question" size={16}/>Emin değilim · Hangisi?</span>
                : <span className="eyebrow">Tanınan işaret</span>}
              <span className="spacer"/>
              {(view === "sure" || view === "unsure") && <span className={`conf-pill ${sure ? "good" : "warn"}`}>{percent(best.confidence)} güven</span>}
            </div>

            {view === "idle" && (
              <div className="result-idle">
                <h3>Tanınan işaret burada belirir</h3>
                <div className="hint-steps">
                  <div><b>1</b>Kendinizi çizgiye yerleştirin</div>
                  <div><b>2</b>{continuous ? "İşareti yapın" : TOUCH ? "Kayıt düğmesine basıp işareti yapın" : "Kayıt düğmesine ya da Boşluk'a basın"}</div>
                  <div><b>3</b>Ellerinizi indirin, sonucu onaylayın</div>
                </div>
              </div>
            )}
            {view === "countdown" && <div className="result-idle"><h3>Hazır olun</h3><p className="muted">Sayım bitince işareti yapın.</p></div>}
            {view === "recording" && (
              <div className="result-idle" style={{width:"100%"}}>
                <h3>İşareti yapın</h3><p className="muted">Bitince ellerinizi indirin; işaret oradan kesilir.</p>
                <div className="meter" style={{width:"100%", maxWidth:320, marginTop:8}}><i style={{width:`${Math.round(live.progress * 100)}%`, animation:"none", transition:"width .1s linear", background:"var(--rec)"}}/></div>
              </div>
            )}
            {view === "reading" && <div className="result-idle"><span className="spinner" style={{width:34, height:34}}/><h3>Okunuyor</h3></div>}
            {view === "error" && (
              <div className="result-idle">
                <div className="note warn"><Icon name="warning" size={20}/>{result.error}. Bir kez daha deneyin.</div>
              </div>
            )}

            {view === "added" && (
              <>
                <div className="word-big reveal" key={`a${resultKey}`}>{added}</div>
                <p className="muted">Cümlenin sonuna eklendi.</p>
                <div className="decide single"><button className="btn btn-secondary" onClick={undo} disabled={!live.running}><Icon name="arrow-counter-clockwise" size={18}/>Geri al</button></div>
              </>
            )}

            {view === "sure" && (
              <>
                <div className={`word-big reveal ${best.text_tr.length > 9 ? "long" : ""}`} key={`w${resultKey}`}>{best.text_tr}</div>
                {result.feedback && <div className="note warn" style={{marginTop:10}}><Icon name="warning" size={19}/>{result.feedback}</div>}
                {pending && (
                  <div className="decide">
                    <button className="btn btn-approve" onClick={() => approve()}><Icon name="check" size={20}/>Doğru{!TOUCH && <kbd>Enter</kbd>}</button>
                    <button className="btn btn-reject" onClick={reject}><Icon name="x" size={20}/>Yanlış{!TOUCH && <kbd>Esc</kbd>}</button>
                  </div>
                )}
                {top.length > 1 && (
                  <div className="alts">
                    <div className="eyebrow">Diğer olasılıklar</div>
                    {top.slice(1).map((p, i) => (
                      <button key={p.gloss} className="alt" onClick={() => approve(p.gloss)} disabled={!live.running} title={`"${p.text_tr}" olarak onayla`}>
                        <strong>{p.text_tr}</strong><Meter value={p.confidence} kind="plain" thin delay={200 + i * 80}/><em className="num">{percent(p.confidence)}</em>
                      </button>
                    ))}
                  </div>
                )}
              </>
            )}

            {view === "unsure" && (
              <>
                <div className="choices stagger" key={`c${resultKey}`}>
                  {top.map((p, i) => (
                    <button key={p.gloss} className="choice" onClick={() => approve(p.gloss)} disabled={!live.running}>
                      <strong>{p.text_tr}</strong><Meter value={p.confidence} kind={i ? "plain" : "warn"} thin delay={i * 80}/><em className="num">{percent(p.confidence)}</em>
                    </button>
                  ))}
                </div>
                {result.feedback && <div className="note warn" style={{marginTop:12}}><Icon name="warning" size={19}/>{result.feedback}</div>}
                <div className="decide single"><button className="btn btn-reject" onClick={reject}><Icon name="x" size={18}/>Hiçbiri{!TOUCH && <kbd>Esc</kbd>}</button></div>
              </>
            )}

            {showHow && (
              <div className="result-foot">
                <button className="btn btn-ghost sm" onClick={() => setHow(view === "added" && !best ? {gloss:added, text:added} : {gloss:best.gloss, text:best.text_tr})}>
                  <Icon name="play-circle" size={18} duo/>Nasıl yapılır?
                </button>
              </div>
            )}
          </section>

          <section className="panel sentence-card" aria-label="Cümle">
            <div className="sentence-head">
              <span className="eyebrow"><Icon name="chat-circle-text" size={16} duo/>Onaylanan cümle</span>
              <span className="spacer"/>
              {words.length > 0 && <span className="chip num">{words.length} kelime</span>}
            </div>
            {words.length ? (
              <div className="chips">
                {words.map((w, i) => {
                  const last = i === words.length - 1;
                  return (
                    <span key={`${i}-${w}-${last ? freshKey : 0}`} className={`word-chip ${last && freshKey ? "fresh" : ""}`}>
                      {w}
                      {last && live.running && <button className="chip-x" onClick={undo} aria-label="Son kelimeyi geri al" title="Geri al"><Icon name="x" size={14}/></button>}
                    </span>
                  );
                })}
              </div>
            ) : <p className="sentence-empty">Onayladığınız işaretler burada sırayla dizilir.</p>}
            <div className="sentence-foot">
              <p className={`sentence-text ${words.length ? "" : "placeholder"}`}>{words.length ? (sentence || words.join(" ")) : "Cümle henüz boş."}</p>
              <div className="sentence-tools">
                <button className="btn btn-primary sm" onClick={() => speak(sentence || words.join(" "), speech.current.rate)} disabled={!words.length}><Icon name="speaker-high" size={18}/>Seslendir</button>
                <button className="btn btn-ghost sm" onClick={undo} disabled={!words.length || !live.running}><Icon name="arrow-counter-clockwise" size={18}/>Geri al</button>
                <button className="btn btn-ghost sm" onClick={() => live.send({action:"reset_sentence"})} disabled={!words.length || !live.running}><Icon name="trash" size={18}/>Temizle</button>
              </div>
            </div>
          </section>
        </div>
      </div>

      {how && <HowToDrawer sign={how} onClose={() => setHow(null)}/>}
    </div>
  );
}

function Setup({live, model, onDone}){
  const [step, setStep] = useState(live.running ? 2 : 1);
  const [test, setTest] = useState(TEST_SIGNS[0]);
  const [outcome, setOutcome] = useState(null);
  const [how, setHow] = useState(null);
  useEffect(() => { if (step === 1 && live.running) setStep(2); }, [live.running]);
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape" && !how) onDone(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [how, onDone]);
  live.listener.current = (data) => {
    if (data.type !== "result") return;
    if (data.error){ setOutcome({error:data.error}); return; }
    const best = data.top?.[0];
    setOutcome(best?.gloss === test.gloss ? {right:true} : {read:best?.text_tr || "başka bir işaret"});
  };
  const line = framingLine(live);
  const f = live.framing;
  const seen = !!f && f.space_below != null;
  const checks = [
    [seen, "Kamera sizi görüyor"],
    [seen && f.space_below >= 1.2, "Elleriniz aşağıdayken de görünüyor"],
    [!!f?.ok, "Uzaklığınız ve yeriniz uygun"]
  ];

  return (
    <Portal>
    <div className="setup" role="dialog" aria-modal="true" aria-label="Kamera kurulumu">
      <div className="setup-in">
        <div className="setup-top">
          <Logo size={36}/>
          <span className="spacer"/>
          <div className="progress" aria-label={`Adım ${step} / 3`}>{[1, 2, 3].map(i => <i key={i} className={i < step ? "done" : i === step ? "now" : ""}/>)}</div>
          <button className="btn btn-ghost sm" onClick={onDone}>Kurulumu atla</button>
        </div>
        <div className="setup-grid">
          <div className="setup-copy rise" key={step}>
            <div className="eyebrow"><Icon name={["video-camera", "frame-corners", "hand-waving"][step - 1]} size={17} duo/>Kamera kurulumu · Adım {step} / 3</div>
            {step === 1 && (
              <>
                <h1>Önce kamerayı ayarlayalım</h1>
                <p className="lead">SignTurk işaretleri, ellerinizi görebildiği sürece tanır. Bu kısa kurulumu yalnızca bir kez yapacaksınız.</p>
                <div className="privacy"><Icon name="lock" size={20} duo/>Görüntünüz kaydedilmez; yalnızca işaretleri okumak için kullanılır. Tarayıcı kamera izni isteyecek.</div>
                <div className="setup-actions">
                  <button className="btn btn-primary lg" onClick={live.start} disabled={live.starting || !model?.ready}>
                    {live.starting ? <span className="spinner"/> : <Icon name="video-camera" size={20}/>}{live.starting ? "Açılıyor" : !model?.ready ? "Model yükleniyor" : "Kamerayı aç"}
                  </button>
                  <button className="btn btn-ghost lg" onClick={onDone}>Şimdilik atla</button>
                </div>
                {live.error && <div className="note bad" role="alert"><Icon name="warning-circle" size={20}/>{live.error}</div>}
              </>
            )}
            {step === 2 && (
              <>
                <h1>Kendinizi çizgiye yerleştirin</h1>
                <p className="lead">Belinizden yukarısı görünmeli; elleriniz aşağıdayken de kadrajda kalmalı. Çizgi ve köşeler maviye dönünce hazırsınız.</p>
                <ul className="checks">
                  {checks.map(([done, text]) => <li key={text} className={done ? "done" : ""}><span className="check-dot"><Icon name="check" size={16}/></span>{text}</li>)}
                </ul>
                <div className="setup-actions">
                  <button className="btn btn-primary lg" onClick={() => setStep(3)} disabled={!live.steady}>Devam<Icon name="arrow-right" size={20}/></button>
                  <button className="btn btn-ghost lg" onClick={() => setStep(3)}>Yine de devam et</button>
                </div>
              </>
            )}
            {step === 3 && (
              <>
                <h1>Bir deneme yapalım</h1>
                <p className="lead">Bir işaret seçin, nasıl yapıldığına bakın. Sonra kaydedin; sayım bitince işareti yapıp ellerinizi indirin.</p>
                <div className="tests" role="group" aria-label="Deneme işareti">
                  {TEST_SIGNS.map(s => <button key={s.gloss} aria-pressed={test.gloss === s.gloss} onClick={() => { setTest(s); setOutcome(null); }}>{s.text}</button>)}
                </div>
                <div><button className="btn btn-secondary" onClick={() => setHow(test)}><Icon name="play" size={18}/>"{test.text}" nasıl yapılır?</button></div>
                <div className="outcome" aria-live="polite">
                  {live.phase === "reading" ? <span className="status"><span className="spinner"/>Okunuyor</span>
                    : outcome?.right ? <div className="note good"><Icon name="check-circle" size={20} duo/>"{test.text}" tanındı. Her şey hazır.</div>
                    : outcome?.read ? <div className="note info"><Icon name="info" size={20}/>"{outcome.read}" olarak okundu. Bir kez daha deneyebilir ya da devam edebilirsiniz.</div>
                    : outcome?.error ? <div className="note warn"><Icon name="warning" size={20}/>{outcome.error}. Bir kez daha deneyin.</div>
                    : null}
                </div>
                <div className="setup-actions" style={{marginTop:0}}>
                  <button className="btn btn-primary lg" onClick={onDone}>Çeviriye geç<Icon name="arrow-right" size={20}/></button>
                  {!outcome?.right && <button className="btn btn-ghost lg" onClick={onDone}>Denemeyi atla</button>}
                </div>
              </>
            )}
          </div>
          <div className="rise" style={{animationDelay:"80ms"}}>
            <Stage live={live} model={model} sample={step === 1}>
              {step === 3 && <RecordButton live={live} onPress={() => { setOutcome(null); live.record(); }}/>}
              <div className="text">
                <strong className={line.tone}><Icon name={line.tone === "good" ? "check-circle" : line.tone === "bad" ? "warning" : "scan"} size={18}/>{line.text}</strong>
                {step === 3 && <span>{{countdown:"Hazırlanın", recording:"Kaydediliyor", reading:"Okunuyor"}[live.phase] || "Kaydetmek için düğmeye basın"}</span>}
              </div>
              <div className="side"><CameraSelect live={live} glass/></div>
            </Stage>
            {step === 1 && <p className="muted" style={{marginTop:14, display:"flex", gap:10, alignItems:"center"}}><Icon name="frame-corners" size={18}/>Kamerayı açınca kendinizi bu çizgiye yerleştireceksiniz.</p>}
          </div>
        </div>
      </div>
      {how && <HowToDrawer sign={how} onClose={() => setHow(null)}/>}
    </div>
    </Portal>
  );
}
