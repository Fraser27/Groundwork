import { login, isAuthEnabled } from '../auth'

export default function Login() {
  if (!isAuthEnabled()) return null

  return (
    <div className="login-shell">
      <div className="login-card">
        <h1>
          <span className="logo-mark">Lex</span>Graph
        </h1>
        <p className="login-sub">
          A governed semantic layer over your documents and your databases, where every fact carries
          the reason it is believed.
        </p>

        <button className="btn btn-primary btn-block" onClick={login} style={{ padding: '10px 20px' }}>
          Sign in
        </button>

        <div className="login-points">
          <div className="login-point">
            <span className="login-point-mark">—</span>
            <span>
              Every claim records how it was reached: declared by a system of record, quoted from a
              document and checked, read into a document by a model, or inferred by a rule.
            </span>
          </div>
          <div className="login-point">
            <span className="login-point-mark">—</span>
            <span>
              Anything a model claimed waits for a human to approve it before it can shape an answer.
            </span>
          </div>
          <div className="login-point">
            <span className="login-point-mark">—</span>
            <span>
              Ask why, and you get the file, the page and the exact words — or the chain of
              reasoning.
            </span>
          </div>
        </div>

        <p className="login-foot">Authenticated via Amazon Cognito</p>
      </div>
    </div>
  )
}
