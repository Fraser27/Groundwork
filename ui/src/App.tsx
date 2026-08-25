import { useEffect, useState, type ReactNode } from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import { api } from './api'
import { useUnitLabel } from './useUnitLabel'
import {
  isAuthEnabled,
  isAuthenticated,
  handleAuthCallback,
  hasPendingAuthCode,
  getUserEmail,
  getTenantId,
  isPlatformAdmin,
  logout,
} from './auth'
import { AdminOnly, Spinner } from './components/Shared'
import Dashboard from './pages/Dashboard'
import Matters from './pages/Matters'
import Documents from './pages/Documents'
import ReviewQueue from './pages/ReviewQueue'
import Entities from './pages/Entities'
import Tables from './pages/Tables'
import TableDetail from './pages/TableDetail'
import Metrics from './pages/Metrics'
import GraphExplorer from './pages/GraphExplorer'
import QueryBuilder from './pages/QueryBuilder'
import Retrieval from './pages/Retrieval'
import Provenance from './pages/Provenance'
import Admin from './pages/Admin'
import Access from './pages/Access'
import Glossary from './pages/Glossary'
import Platform from './pages/Platform'
import Login from './pages/Login'

function App() {
  const tenant = getTenantId()
  const unit = useUnitLabel()
  const [graphStatus, setGraphStatus] = useState<'connected' | 'disconnected'>('disconnected')
  const [pendingCount, setPendingCount] = useState<number | null>(null)
  const [isHomeTenant, setIsHomeTenant] = useState(false)
  // The code grant needs a round trip to Cognito's token endpoint, so this cannot resolve
  // in a state initialiser the way reading tokens from the URL fragment could.
  //
  // Three states, not two: `null` means "a code is still being exchanged". Rendering the
  // login page during that would flash it at a user who is already part-way through
  // signing in.
  const [authed, setAuthed] = useState<boolean | null>(() =>
    hasPendingAuthCode() ? null : isAuthenticated(),
  )

  useEffect(() => {
    if (authed !== null) return
    handleAuthCallback()
      .then((ok) => setAuthed(ok || isAuthenticated()))
      .catch(() => setAuthed(false))
  }, [authed])
  const [theme, setTheme] = useState<'light' | 'dark'>(
    () => (localStorage.getItem('theme') as 'light' | 'dark') || 'light',
  )
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem('sidebarCollapsed') === 'true')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  useEffect(() => {
    localStorage.setItem('sidebarCollapsed', String(collapsed))
  }, [collapsed])

  useEffect(() => {
    if (!authed) return
    api
      .health()
      .then((h) => {
        setGraphStatus(h.graph === 'connected' ? 'connected' : 'disconnected')
        // Only the operator tenant gets the platform screen. The routes enforce this anyway,
        // so this hides a nav item rather than granting anything.
        setIsHomeTenant(!!h.home_tenant && h.home_tenant === tenant)
      })
      .catch(() => setGraphStatus('disconnected'))
    // The pending badge is the one number worth carrying in the chrome: it is a
    // queue of claims nobody has signed off yet.
    api
      .dashboard(tenant)
      .then((d) => setPendingCount(d.pending_review))
      .catch(() => setPendingCount(null))
  }, [authed, tenant])

  if (isAuthEnabled() && authed === null) return <Spinner />
  if (isAuthEnabled() && !authed) return <Login />

  const admin = isPlatformAdmin()
  // The route stays `/matters`: it is the scoping key's name, and a URL is not user-facing copy.
  const unitPlural = unit.plural

  return (
    <div className={`app-layout${collapsed ? ' sidebar-collapsed' : ''}`}>
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="sidebar-logo-text">
            <h1>
              <span className="logo-mark">Ground</span>work
            </h1>
            <span>Every fact carries its provenance</span>
          </div>
          <button
            className="sidebar-toggle"
            onClick={() => setCollapsed((c) => !c)}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? '»' : '«'}
          </button>
        </div>

        <nav>
          <NavItem to="/" end icon={icons.dashboard} label="Dashboard" collapsed={collapsed} />
          {/* Ask is deprecated and no longer in the menu. Retrieval renders the same trace, the
              same evidence panels and the same wall, over an agent loop rather than one request.
              The route stays so a bookmark still resolves, and because the page is the reference
              the Retrieval panels were extracted from. */}
          <NavItem to="/retrieval" icon={icons.retrieval} label="Retrieval" collapsed={collapsed} />
          <NavItem
            to="/review"
            icon={icons.review}
            label="Review queue"
            collapsed={collapsed}
            count={pendingCount ?? undefined}
          />

          <div className="nav-section">Knowledge</div>
          <div className="nav-section-rule" />
          <NavItem to="/matters" icon={icons.matters} label={unitPlural} collapsed={collapsed} />
          <NavItem to="/documents" icon={icons.documents} label="Documents" collapsed={collapsed} />
          <NavItem to="/entities" icon={icons.entities} label="Entities" collapsed={collapsed} />
          <NavItem to="/graph" icon={icons.graph} label="Graph" collapsed={collapsed} />
          <NavItem to="/provenance" icon={icons.provenance} label="Audit" collapsed={collapsed} />

          <div className="nav-section">Structured data</div>
          <div className="nav-section-rule" />
          <NavItem to="/tables" icon={icons.tables} label="Tables" collapsed={collapsed} />
          <NavItem to="/metrics" icon={icons.metrics} label="Metrics" collapsed={collapsed} />

          {admin && (
            <>
              <div className="nav-section">Administration</div>
              <div className="nav-section-rule" />
              <NavItem to="/glossary" icon={icons.glossary} label="Glossary" collapsed={collapsed} />
              <NavItem to="/access" icon={icons.access} label="Access" collapsed={collapsed} />
              <NavItem to="/admin" icon={icons.admin} label="Admin" collapsed={collapsed} />
              {isHomeTenant && (
                <NavItem
                  to="/platform"
                  icon={icons.platform}
                  label="Platform"
                  collapsed={collapsed}
                />
              )}
            </>
          )}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-tenant" title={`Tenant: ${tenant}`}>
            Tenant <strong>{tenant}</strong>
          </div>
          {isAuthEnabled() && (
            <div className="sidebar-user">
              <span title={getUserEmail()}>{getUserEmail()}</span>
              <button onClick={logout} className="sidebar-signout">
                Sign out
              </button>
            </div>
          )}
          <button
            className="theme-toggle"
            onClick={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
            title={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
          >
            <span className="theme-toggle-icon" aria-hidden="true">
              {theme === 'light' ? '☾' : '☀'}
            </span>
            <span className="nav-label">{theme === 'light' ? 'Dark mode' : 'Light mode'}</span>
          </button>
          <div className="sidebar-status" title={`Graph: ${graphStatus}`}>
            <span className={`status-dot ${graphStatus}`} />
            <span className="nav-label">Graph: {graphStatus}</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          {/* Deprecated, and out of the menu. Kept routed so an existing bookmark resolves. */}
          <Route path="/query" element={<QueryBuilder />} />
          <Route path="/retrieval" element={<Retrieval />} />
          <Route path="/review" element={<ReviewQueue />} />
          <Route path="/matters" element={<Matters />} />
          <Route path="/documents" element={<Documents />} />
          <Route path="/documents/:id" element={<Documents />} />
          <Route path="/entities" element={<Entities />} />
          <Route path="/graph" element={<GraphExplorer />} />
          <Route path="/provenance" element={<Provenance />} />
          <Route path="/tables" element={<Tables />} />
          <Route path="/tables/:name" element={<TableDetail />} />
          <Route path="/metrics" element={<Metrics />} />
          <Route path="/glossary" element={<Glossary />} />
          <Route
            path="/access"
            element={
              <AdminOnly>
                <Access />
              </AdminOnly>
            }
          />
          <Route
            path="/admin"
            element={
              <AdminOnly>
                <Admin />
              </AdminOnly>
            }
          />
          <Route
            path="/platform"
            element={
              <AdminOnly>
                <Platform />
              </AdminOnly>
            }
          />
        </Routes>
      </main>
    </div>
  )
}

function NavItem({
  to,
  icon,
  label,
  collapsed,
  end,
  count,
}: {
  to: string
  icon: ReactNode
  label: string
  collapsed: boolean
  end?: boolean
  count?: number
}) {
  return (
    <NavLink to={to} end={end} title={collapsed ? label : undefined}>
      <span className="nav-icon">{icon}</span>
      <span className="nav-label">{label}</span>
      {!!count && <span className="nav-count">{count}</span>}
    </NavLink>
  )
}

const svg = (path: ReactNode) => (
  <svg
    viewBox="0 0 24 24"
    width="17"
    height="17"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.9"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    {path}
  </svg>
)

const icons = {
  dashboard: svg(
    <>
      <rect x="3" y="3" width="7" height="7" />
      <rect x="14" y="3" width="7" height="7" />
      <rect x="14" y="14" width="7" height="7" />
      <rect x="3" y="14" width="7" height="7" />
    </>,
  ),
  review: svg(
    <>
      <path d="M9 11l3 3L22 4" />
      <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
    </>,
  ),
  matters: svg(
    <>
      <rect x="2" y="7" width="20" height="14" rx="2" />
      <path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M2 13h20" />
    </>,
  ),
  documents: svg(
    <>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6M8 13h8M8 17h5" />
    </>,
  ),
  entities: svg(
    <>
      <circle cx="8" cy="12" r="5" />
      <circle cx="16" cy="12" r="5" />
    </>,
  ),
  retrieval: svg(
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
    </>,
  ),
  platform: svg(
    <>
      <rect x="3" y="4" width="18" height="6" rx="1.5" />
      <rect x="3" y="14" width="18" height="6" rx="1.5" />
    </>,
  ),
  graph: svg(
    <>
      <circle cx="5" cy="6" r="2.5" />
      <circle cx="19" cy="6" r="2.5" />
      <circle cx="12" cy="18" r="2.5" />
      <path d="M7 7l3.5 8.5M17 7l-3.5 8.5M7 6h10" />
    </>,
  ),
  provenance: svg(
    <>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <path d="M9 12l2 2 4-4" />
    </>,
  ),
  tables: svg(
    <>
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M3 9h18M3 15h18M9 3v18" />
    </>,
  ),
  metrics: svg(
    <>
      <path d="M3 3v18h18" />
      <path d="M7 14l4-4 3 3 5-6" />
    </>,
  ),
  glossary: svg(
    <>
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
      <path d="M9 7h7M9 11h5" />
    </>,
  ),
  access: svg(
    <>
      <rect x="4" y="10.5" width="16" height="10.5" rx="2" />
      <path d="M8.5 10.5V7a3.5 3.5 0 0 1 7 0v3.5M12 14.5v2.5" />
    </>,
  ),
  admin: svg(
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </>,
  ),
}

export default App
