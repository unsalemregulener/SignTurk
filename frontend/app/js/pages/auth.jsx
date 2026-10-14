// Sign in and sign up: the form on the left, a camera stage with the avatar on the right.

function AuthScreen({onLogin}){
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({full_name:"", email:"", password:""});
  const [reveal, setReveal] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [ok, setOk] = useState("");
  const login = mode === "login";

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true); setError(""); setOk("");
    try{
      if (login){
        const data = await apiFetch(API.login, {method:"POST", body:JSON.stringify({email:form.email, password:form.password})});
        onLogin({...data.user, access_token:data.access_token});
      }else{
        await apiFetch(API.register, {method:"POST", body:JSON.stringify(form)});
        setOk("Hesabınız oluşturuldu. Şimdi giriş yapabilirsiniz.");
        setMode("login");
      }
    }catch(err){ setError(err.message || "İşlem tamamlanamadı"); }
    finally{ setBusy(false); }
  };
  const switchTo = (next) => { setMode(next); setError(""); setOk(""); };
  const set = (key) => (e) => setForm({...form, [key]:e.target.value});

  return (
    <div className="auth">
      <section className="auth-side">
        <Logo size={40}/>
        <div className="auth-card rise" key={mode}>
          <h1>{login ? "Tekrar hoş geldiniz" : "Hesap oluşturun"}</h1>
          <p className="sub">{login ? "Çeviriye devam etmek için giriş yapın." : "Geçmişiniz ve ayarlarınız hesabınızda saklanır."}</p>
          <Seg label="Giriş ya da kayıt" value={mode} onChange={switchTo}
            options={[{value:"login", label:"Giriş yap"}, {value:"register", label:"Kayıt ol"}]}/>
          <form onSubmit={submit} style={{marginTop:16}}>
            {error && <div className="note bad" role="alert"><Icon name="warning-circle" size={20}/>{error}</div>}
            {ok && <div className="note good" role="status"><Icon name="check-circle" size={20}/>{ok}</div>}
            {!login && (
              <div className="field"><label htmlFor="name">Ad soyad</label>
                <div className="input-wrap"><Icon name="user-circle" size={20}/>
                  <input id="name" className="input" value={form.full_name} onChange={set("full_name")} required autoComplete="name"/></div></div>
            )}
            <div className="field"><label htmlFor="email">E-posta</label>
              <div className="input-wrap"><Icon name="envelope" size={20}/>
                <input id="email" className="input" type="email" value={form.email} onChange={set("email")} required autoComplete="email" placeholder="ornek@posta.com"/></div></div>
            <div className="field"><label htmlFor="password">Şifre</label>
              <div className="input-wrap"><Icon name="lock" size={20}/>
                <input id="password" className="input" type={reveal ? "text" : "password"} value={form.password} onChange={set("password")} required
                  autoComplete={login ? "current-password" : "new-password"} minLength={login ? undefined : 8} style={{paddingRight:52}}
                  placeholder={login ? "" : "En az 8 karakter"}/>
                <span className="end"><button type="button" className="btn btn-ghost sm icon-only" onClick={() => setReveal(v => !v)}
                  aria-label={reveal ? "Şifreyi gizle" : "Şifreyi göster"}><Icon name={reveal ? "eye-slash" : "eye"} size={19}/></button></span>
              </div></div>
            <button className="btn btn-primary lg block" type="submit" disabled={busy} style={{marginTop:6}}>
              {busy ? <span className="spinner"/> : <Icon name={login ? "sign-in" : "user-plus"} size={20}/>}{login ? "Giriş yap" : "Hesap oluştur"}
            </button>
          </form>
        </div>
        <div className="auth-foot"><Icon name="lock" size={17}/>Kamera görüntüsü kaydedilmez; yalnızca onayladığınız kelimeler saklanır.</div>
      </section>

      <section className="auth-brand stage-bg on-stage">
        <BrandArt/>
        <h2>El hareketinden <em>kelimeye</em>, kameranın önünde.</h2>
        <p className="lead">SignTurk Türk İşaret Dili'ni kameradan tanır; onayladığınız kelimelerden Türkçe cümle kurar.</p>
        <div className="auth-show" aria-hidden="true">
          <FrameBox state="ok"/>
          <img className="auth-avatar" src={AVATAR_POSES.wave} alt="" onError={e => { e.currentTarget.style.display = "none"; }}/>
          <div className="demo-card glass">
            <div><span className="eyebrow">Tanınan işaret</span><strong>MERHABA</strong></div>
            <span className="conf-pill on-stage">%98 güven</span>
            <span className="btn btn-approve"><Icon name="check" size={18}/>Doğru</span>
          </div>
        </div>
        <div className="auth-facts">
          <div><b>226</b><span>işaret</span></div>
          <div><b>%94</b><span>AUTSL test doğruluğu</span></div>
          <div><b>4 sn</b><span>bir kayıt</span></div>
        </div>
      </section>
    </div>
  );
}
