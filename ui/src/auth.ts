/**
 * Cognito authentication helpers.
 *
 * Uses the Cognito Hosted UI with **authorization code + PKCE**, not the implicit
 * flow. The app client permits only the code grant (`auth-stack.ts`), because implicit
 * returns tokens in the URL fragment where browser history keeps them — not somewhere a
 * token granting access to privileged documents belongs. Asking for `response_type=token`
 * against that client fails with `unauthorized_client`.
 *
 * PKCE rather than a client secret: this is a public client shipped to browsers, so a
 * secret would be readable by anyone who opens devtools. The verifier is held in session
 * storage only between redirect and token exchange.
 *
 * Config comes from /runtime-config.json (written by CDK at deploy time), falling back
 * to Vite env vars for local dev. When no config is present at all, auth is treated as
 * disabled so the UI is usable against a local API.
 */

interface RuntimeConfig {
  cognitoUserPoolId?: string
  cognitoClientId?: string
  cognitoRegion?: string
  cognitoDomain?: string
  defaultTenantId?: string
}

let runtimeConfig: RuntimeConfig = {}

export async function loadRuntimeConfig(): Promise<void> {
  try {
    const res = await fetch('/runtime-config.json')
    if (res.ok) runtimeConfig = await res.json()
  } catch {
    // No runtime config — fall back to Vite env vars (local dev)
  }
}

function cfg(runtimeKey: keyof RuntimeConfig, viteKey: string): string {
  return runtimeConfig[runtimeKey] || import.meta.env[viteKey] || ''
}

const getPoolId = () => cfg('cognitoUserPoolId', 'VITE_COGNITO_USER_POOL_ID')
const getClientId = () => cfg('cognitoClientId', 'VITE_COGNITO_CLIENT_ID')

/**
 * A bare host, scheme stripped if one was supplied. Callers prepend `https://`.
 *
 * The stack output `HostedUiDomain` carries a scheme while the config key wants a host, so
 * pasting the output into runtime-config.json yields `https://https//...` and the browser
 * tries to resolve a host named `https`. Cheaper to accept both than to be right about
 * which one is in the file.
 */
const getDomain = () =>
  cfg('cognitoDomain', 'VITE_COGNITO_DOMAIN')
    .replace(/^https?:\/\//, '')
    .replace(/\/+$/, '')

export function isAuthEnabled(): boolean {
  return !!(getPoolId() && getClientId() && getDomain())
}

/**
 * The tenant the session belongs to. Read from the ID token claim where present:
 * the server enforces tenancy from the verified JWT regardless, so this is only
 * used to build request paths.
 */
export function getTenantId(): string {
  return (
    localStorage.getItem('tenant_id') ||
    cfg('defaultTenantId', 'VITE_DEFAULT_TENANT_ID') ||
    'demo-firm'
  )
}

export function setTenantId(tenantId: string): void {
  localStorage.setItem('tenant_id', tenantId)
}

function getRedirectUri(): string {
  return `${window.location.origin}/`
}

const PKCE_VERIFIER_KEY = 'pkce_verifier'

/** True when this page load is a Cognito redirect carrying an authorization code. */
export function hasPendingAuthCode(): boolean {
  return new URLSearchParams(window.location.search).has('code')
}

function base64Url(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '')
}

async function createPkcePair(): Promise<{ verifier: string; challenge: string }> {
  const verifier = base64Url(crypto.getRandomValues(new Uint8Array(32)))
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier))
  return { verifier, challenge: base64Url(new Uint8Array(digest)) }
}

export async function login(): Promise<void> {
  if (!isAuthEnabled()) return
  const { verifier, challenge } = await createPkcePair()
  // sessionStorage, not localStorage: the verifier is single-use and belongs to this tab's
  // login attempt only.
  sessionStorage.setItem(PKCE_VERIFIER_KEY, verifier)

  const params = new URLSearchParams({
    client_id: getClientId(),
    response_type: 'code',
    scope: 'openid email profile',
    redirect_uri: getRedirectUri(),
    code_challenge_method: 'S256',
    code_challenge: challenge,
  })
  window.location.href = `https://${getDomain()}/login?${params}`
}

export function logout(): void {
  clearTokens()
  if (!isAuthEnabled()) {
    window.location.reload()
    return
  }
  const params = new URLSearchParams({
    client_id: getClientId(),
    logout_uri: getRedirectUri(),
  })
  window.location.href = `https://${getDomain()}/logout?${params}`
}

/**
 * Exchange the `?code=` from the Cognito redirect for tokens.
 *
 * Async because the code grant needs a round trip to the token endpoint — the implicit
 * flow handed tokens back in the fragment, which is exactly the property that made it
 * unsuitable.
 */
export async function handleAuthCallback(): Promise<boolean> {
  const query = new URLSearchParams(window.location.search)
  const code = query.get('code')
  if (!code) return false

  const verifier = sessionStorage.getItem(PKCE_VERIFIER_KEY)
  if (!verifier) {
    // A code with no verifier means this is a stale or replayed redirect. Clear it rather
    // than attempting an exchange that cannot succeed.
    window.history.replaceState(null, '', '/')
    return false
  }

  try {
    const res = await fetch(`https://${getDomain()}/oauth2/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'authorization_code',
        client_id: getClientId(),
        code,
        redirect_uri: getRedirectUri(),
        code_verifier: verifier,
      }),
    })
    if (!res.ok) return false

    const tokens = await res.json()
    if (!tokens.id_token || !tokens.access_token) return false

    localStorage.setItem('id_token', tokens.id_token)
    localStorage.setItem('access_token', tokens.access_token)
    localStorage.setItem('token_expiry', String(Date.now() + (tokens.expires_in ?? 3600) * 1000))
    if (tokens.refresh_token) localStorage.setItem('refresh_token', tokens.refresh_token)

    const payload = decodeClaims(tokens.id_token)
    if (payload) {
      localStorage.setItem(
        'user_email',
        String(payload.email || payload['cognito:username'] || 'user'),
      )
      // Tenant is a token claim, not a user preference: a caller must not be able
      // to widen their own scope by editing it.
      const tenant = payload['custom:tenant_id'] || payload.tenant_id
      if (tenant) localStorage.setItem('tenant_id', String(tenant))
    } else {
      localStorage.setItem('user_email', 'user')
    }
    return true
  } finally {
    // Single-use, and the code must not survive in the address bar either way.
    sessionStorage.removeItem(PKCE_VERIFIER_KEY)
    window.history.replaceState(null, '', '/')
  }
}

function decodeClaims(idToken: string): Record<string, unknown> | null {
  try {
    return JSON.parse(atob(idToken.split('.')[1]))
  } catch {
    return null
  }
}

function idTokenClaims(): Record<string, unknown> | null {
  const token = localStorage.getItem('id_token')
  return token ? decodeClaims(token) : null
}

export function getAccessToken(): string | null {
  if (!isAuthEnabled()) return null

  const token = localStorage.getItem('access_token')
  const expiry = parseInt(localStorage.getItem('token_expiry') || '0')

  if (!token || Date.now() > expiry) {
    clearTokens()
    return null
  }
  return token
}

export function isAuthenticated(): boolean {
  if (!isAuthEnabled()) return true // Auth disabled in local dev
  return !!getAccessToken()
}

export function getUserEmail(): string {
  return localStorage.getItem('user_email') || 'user'
}

/** Role names from the token, used only to hide UI, never to authorise. */
export function getUserRoles(): string[] {
  const claims = idTokenClaims()
  // Roles are Cognito groups. `custom:roles` is read too so a non-Cognito issuer works.
  const groups = claims?.['cognito:groups'] ?? claims?.['custom:roles']
  if (Array.isArray(groups)) return groups.map(String).filter(Boolean)
  if (typeof groups === 'string') {
    return groups
      .split(',')
      .map((r) => r.trim())
      .filter(Boolean)
  }
  return []
}

/**
 * Whether to render administration controls. Not a security boundary: `require_admin`
 * in `src/api/deps.py` answers 403 whatever the browser chose to draw.
 */
export function isPlatformAdmin(): boolean {
  if (!isAuthEnabled()) return true
  return getUserRoles().includes('platform-admin')
}

function clearTokens(): void {
  localStorage.removeItem('id_token')
  localStorage.removeItem('access_token')
  localStorage.removeItem('token_expiry')
  localStorage.removeItem('user_email')
}
