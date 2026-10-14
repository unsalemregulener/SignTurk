// The app frame: navigation rail (tab bar on phones), routing and page transitions.

const ROUTES = [
  {id:"home", label:"Ana sayfa", icon:"house"},
  {id:"live", label:"Çeviri", icon:"video-camera"},
  {id:"dictionary", label:"Sözlük", icon:"book-open-text"},
  {id:"history", label:"Geçmiş", icon:"clock-counter-clockwise"}
];
const LEGACY_ROUTES = {dashboard:"home", avatar3d:"dictionary", admin:"admin/overview"};

function parseRoute(){
  const raw = window.location.hash.replace(/^#\/?/, "") || "home";
  return LEGACY_ROUTES[raw] || raw;
}

// Swap pages inside a view transition where the browser has one.
function withTransition(update){
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!document.startViewTransition || reduce){ update(); return; }
  document.startViewTransition(() => ReactDOM.flushSync(update));
}

function UserMenu({user, isAdmin, navigate, onLogout, onClose, style}){
  const [theme, setTheme] = useTheme();
  useEffect(() => {
    const close = () => onClose();
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("click", close);
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("click", close); window.removeEventListener("keydown", onKey); };
  }, [onClose]);
  const nextTheme = {system:"light", light:"dark", dark:"system"}[theme];
  const themeName = {system:"Sistem", light:"Açık", dark:"Koyu"};
  return (
    <div className="menu" role="menu" style={style} onClick={e => e.stopPropagation()}>
      <div className="menu-head"><span className="initials">{initials(user)}</span><div><strong>{user?.full_name || "Kullanıcı"}</strong><span>{user?.email}</span></div></div>
      <button role="menuitem" onClick={() => { navigate("settings"); onClose(); }}><Icon name="gear-six" size={19}/>Ayarlar</button>
      {isAdmin && <button role="menuitem" onClick={() => { navigate("admin/overview"); onClose(); }}><Icon name="shield-check" size={19}/>Yönetim</button>}
      <button role="menuitem" onClick={() => setTheme(nextTheme)}>
        <Icon name={theme === "dark" ? "moon" : theme === "light" ? "sun" : "monitor"} size={19}/>Görünüm: {themeName[theme]}
      </button>
      <button role="menuitem" onClick={onLogout} style={{color:"var(--bad)"}}><Icon name="sign-out" size={19}/>Çıkış yap</button>
    </div>
  );
}

function Shell({user, onLogout}){
  const [route, setRoute] = useState(parseRoute);
  const [menu, setMenu] = useState(null);   // "rail" or "top" while open
  const model = useHealth();
  const isAdmin = (user?.role || "").toLowerCase() === "admin";

  useEffect(() => {
    const onHash = () => { withTransition(() => setRoute(parseRoute())); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const navigate = useCallback((id) => { window.location.hash = "#/" + id; }, []);
  const closeMenu = useCallback(() => setMenu(null), []);

  const known = ["home", "live", "dictionary", "history", "settings"].includes(route) || (isAdmin && route.startsWith("admin/"));
  const active = known ? route : "home";
  const section = active.startsWith("admin/") ? "admin" : active;
  let page;
  if (active === "live") page = <TranslatePage model={model} onExpired={onLogout}/>;
  else if (active === "dictionary") page = <DictionaryPage navigate={navigate}/>;
  else if (active === "history") page = <HistoryPage navigate={navigate}/>;
  else if (active === "settings") page = <SettingsPage user={user} navigate={navigate} onLogout={onLogout}/>;
  else if (section === "admin") page = <AdminPage tab={active} navigate={navigate} model={model}/>;
  else page = <HomePage user={user} model={model} navigate={navigate}/>;

  const railItem = (id, label, icon, href) => (
    <a key={id} className="rail-item" href={`#/${href || id}`} aria-current={section === id ? "page" : undefined}>
      <span className="icon-box"><Icon name={icon} size={22} duo={section === id}/></span>{label}
    </a>
  );

  return (
    <div className="app">
      <nav className="rail" aria-label="Ana menü">
        <a className="rail-logo" href="#/home" aria-label="SignTurk ana sayfa"><Mark size={46}/></a>
        {ROUTES.map(r => railItem(r.id, r.label, r.icon))}
        <span className="rail-spacer"/>
        {isAdmin && railItem("admin", "Yönetim", "shield-check", "admin/overview")}
        {railItem("settings", "Ayarlar", "gear-six")}
        <div style={{position:"relative"}}>
          <button className="rail-user" onClick={e => { e.stopPropagation(); setMenu(menu ? null : "rail"); }} aria-haspopup="menu" aria-expanded={menu === "rail"} aria-label="Hesap menüsü">
            <span className="initials">{initials(user)}</span>
          </button>
          {menu === "rail" && <UserMenu user={user} isAdmin={isAdmin} navigate={navigate} onLogout={onLogout} onClose={closeMenu} style={{left:58, bottom:0}}/>}
        </div>
      </nav>

      <main className="main">
        <div className="mobile-top">
          <Logo size={34}/>
          <span className="spacer"/>
          <div style={{position:"relative"}}>
            <button className="rail-user" onClick={e => { e.stopPropagation(); setMenu(menu ? null : "top"); }} aria-haspopup="menu" aria-expanded={menu === "top"} aria-label="Hesap menüsü">
              <span className="initials">{initials(user)}</span>
            </button>
            {menu === "top" && <UserMenu user={user} isAdmin={isAdmin} navigate={navigate} onLogout={onLogout} onClose={closeMenu} style={{right:0, top:52, transformOrigin:"top right"}}/>}
          </div>
        </div>
        {page}
      </main>

      <nav className="tabbar" aria-label="Sayfalar">
        {[...ROUTES, {id:"settings", label:"Ayarlar", icon:"gear-six"}].map(r => (
          <a key={r.id} href={`#/${r.id}`} aria-current={section === r.id ? "page" : undefined}>
            <Icon name={r.icon} size={23} duo={section === r.id}/>{r.label}
          </a>
        ))}
      </nav>
    </div>
  );
}

function App(){
  const [user, setUserState] = useState(() => { const saved = readJSON(KEYS.user); return saved?.access_token ? saved : null; });
  const setUser = useCallback((next) => {
    if (next) writeKey(KEYS.user, JSON.stringify(next));
    else{
      if (storedToken()) apiFetch(API.logout, {method:"POST"}).catch(() => {});
      writeKey(KEYS.user, null);
    }
    setUserState(next);
  }, []);
  useEffect(() => {
    const onLogout = () => setUser(null);
    window.addEventListener("signturk:logout", onLogout);
    return () => window.removeEventListener("signturk:logout", onLogout);
  }, [setUser]);
  const logout = useCallback(() => setUser(null), [setUser]);
  return user ? <Shell user={user} onLogout={logout}/> : <AuthScreen onLogin={setUser}/>;
}

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
