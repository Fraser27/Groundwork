/**
 * ReasonerReportPanel — why a rule check drew nothing.
 *
 * Zero conclusions has four causes and they are not interchangeable. A rule that ran over the
 * facts and found no conflict is a clean check. A rule whose premises matched nothing never
 * checked anything, and reads identically unless it is said out loud — which is the failure the
 * whole design is organised against. So the four states are drawn apart, never summed:
 *
 *   cleared   ran, joined, concluded nothing        the only one that is reassurance
 *   starved   ran, and a named premise emptied it   the actionable one; the premise is quoted
 *   skipped   could never fire at all               the check does not exist
 *   refused   fired, and an invariant rejected it   a match was found and dropped
 *
 * Counts go beside them because "0 over 40 facts with 3 rules evaluated" and "0 because nothing
 * ran" are different claims about the system, and the number is the only thing separating them.
 */

import type { CSSProperties, ReactNode } from 'react'
import type { ReasonerReport, RuleDef } from '../api'
import FieldHelp from './FieldHelp'

/** `conclusions_refused` entries are `${rule_id}: ${error}`. Split so the rule can be named. */
function splitRefusal(line: string): { ruleId: string; detail: string } {
  const at = line.indexOf(': ')
  if (at < 0) return { ruleId: line, detail: '' }
  return { ruleId: line.slice(0, at), detail: line.slice(at + 2) }
}

export default function ReasonerReportPanel({
  report,
  rules = [],
}: {
  report: ReasonerReport
  /** The pack's rules, so the ones that ran and found nothing can be named rather than counted. */
  rules?: RuleDef[]
}) {
  const starved = Object.entries(report.rules_starved ?? {})
  const skipped = Object.entries(report.rules_skipped ?? {})
  const refused = (report.conclusions_refused ?? []).map(splitRefusal)

  // By rule, not by line: `conclusions_refused` is one entry per refused conclusion, so summing
  // the three lists would report more rules than the pack has.
  const unable = new Set([
    ...starved.map(([id]) => id),
    ...skipped.map(([id]) => id),
    ...refused.map((r) => r.ruleId),
  ])
  const concluded = new Set(report.inferences.map((i) => i.rule_id))
  const cleared = rules.filter((r) => !unable.has(r.id) && !concluded.has(r.id))
  const drew = report.count > 0

  const scale = `${report.rules_evaluated} ${report.rules_evaluated === 1 ? 'rule' : 'rules'} over ${
    report.facts_considered
  } ${report.facts_considered === 1 ? 'fact' : 'facts'}`

  return (
    <>
      {/* First, and never folded away. A reader who stops at the counts below must already have
          been told whether they are looking at a clean check or an unperformed one. */}
      {drew ? (
        <div className="banner banner-info">
          <span>
            <strong>
              {report.count} {report.count === 1 ? 'conclusion' : 'conclusions'} staged for review.
            </strong>{' '}
            Drawn from {scale}. Nothing was published, each one carries the facts it rests on, so
            it can be followed back to the documents underneath before anyone acts on it.
            {unable.size > 0 &&
              ` ${unable.size} other ${unable.size === 1 ? 'rule' : 'rules'} still drew nothing for a reason worth reading below.`}
          </span>
        </div>
      ) : report.rules_evaluated === 0 && unable.size === 0 ? (
        // Not green, and not "clean". A pack carrying no rules performs no checks, and the
        // reassuring reading of zero must not be available here.
        <div className="banner banner-warn">
          <span>
            <strong>No rule ran, because this pack defines none.</strong> Nothing was checked for
            conflicts, stale authority or anything else. This is not a clean result; it is the
            absence of a check.
          </span>
        </div>
      ) : unable.size === 0 ? (
        <div className="banner banner-info">
          <span>
            <strong>Checked, and nothing follows.</strong> {scale}, and every premise joined. No
            conflict, no stale authority, nothing else the pack looks for. Read this as a clean
            result from a check that actually ran.
          </span>
        </div>
      ) : (
        <div className="banner banner-warn">
          <span>
            <strong>Nothing was concluded, and not every rule was able to check.</strong> {scale}.{' '}
            {unable.size} of them drew nothing because they could not, rather than because they
            found nothing. A conflict check with no adversity facts to join returns exactly what a
            clean one returns, so each is named below with the premise that emptied it.
            <FieldHelp
              title="Why this distinction is drawn"
              text="A conflict check that finds no conflict and a conflict check that had nothing to look at both report zero. Only one of them is reassurance. The rule and the premise that emptied it are named so you can tell which you are reading."
            />
          </span>
        </div>
      )}

      <dl className="qtrace-facts" style={{ marginTop: 14 }}>
        <div>
          <dt>Rules evaluated</dt>
          <dd>{report.rules_evaluated}</dd>
        </div>
        <div>
          <dt>
            Facts considered
            <FieldHelp text="The tenant's live, signed-off facts. A rule may only rest on facts somebody stands behind, so a pending or withdrawn fact is not among these." />
          </dt>
          <dd>{report.facts_considered}</dd>
        </div>
        <div>
          <dt>Conclusions drawn</dt>
          <dd className={drew ? 'qtrace-cleared' : undefined}>{report.count}</dd>
        </div>
        <div>
          <dt>
            Could not check
            <FieldHelp text="Rules that returned nothing for a reason other than finding nothing: the join was empty, the rule could never fire, or an invariant refused what it concluded." />
          </dt>
          <dd className={unable.size > 0 ? 'qtrace-withheld' : undefined}>{unable.size}</dd>
        </div>
      </dl>

      {starved.length > 0 && (
        <RuleGroup
          // Orange, not red: the rule is fine and a fact is missing. Distinct from the two red
          // groups below, where the check itself is broken.
          tone="orange"
          title={`${starved.length} ${starved.length === 1 ? 'rule' : 'rules'} ran and found no facts to join`}
          tag={<span className="tag tag-orange">join empty</span>}
          note={
            'The rule was evaluated against every live fact and one of its premises matched nothing, ' +
            'so it never reached a conclusion either way. The missing premise is named verbatim: it ' +
            'is the thing to fix, and it is usually a relationship nobody has recorded yet rather ' +
            'than a rule that is wrong.'
          }
          rows={starved.map(([ruleId, reason]) => ({
            ruleId,
            label: 'Premise that emptied the join',
            detail: reason,
            rule: rules.find((r) => r.id === ruleId),
          }))}
        />
      )}

      {skipped.length > 0 && (
        <RuleGroup
          title={`${skipped.length} ${skipped.length === 1 ? 'rule' : 'rules'} could not fire at all`}
          tag={<span className="tag tag-red">never ran</span>}
          note={
            'Not the same as an empty join. These were never evaluated, the rule could not be read, ' +
            'or it concludes a relationship outside this pack’s vocabulary. Whatever they were ' +
            'meant to check is not being checked by anything.'
          }
          rows={skipped.map(([ruleId, reason]) => ({
            ruleId,
            label: 'Why it cannot run',
            detail: reason,
            rule: rules.find((r) => r.id === ruleId),
          }))}
        />
      )}

      {refused.length > 0 && (
        <RuleGroup
          title={`${refused.length} ${refused.length === 1 ? 'conclusion' : 'conclusions'} refused by an invariant`}
          tag={<span className="tag tag-red">refused</span>}
          note={
            'These rules did find a match. The conclusion was then rejected when it was written, ' +
            'most often because a variable bound to the wrong kind of entity. Something was found ' +
            'and dropped, which is why it is reported rather than only logged.'
          }
          rows={refused.map(({ ruleId, detail }, i) => ({
            ruleId,
            key: `${ruleId}-${i}`,
            label: 'Reason it was refused',
            detail,
            rule: rules.find((r) => r.id === ruleId),
          }))}
        />
      )}

      {cleared.length > 0 ? (
        <>
          <div className="qtrace-sublabel" style={{ marginTop: 16 }}>
            Checked and clean
            <FieldHelp text="These rules were evaluated, every premise matched, and nothing followed. This is the only part of this report that is reassurance, and it is listed by name so that a rule quietly missing from it stands out." />
          </div>
          <ul className="qtrace-list">
            {cleared.map((r) => (
              <li key={r.id}>
                <strong>{r.id}</strong>: {r.description}
              </li>
            ))}
          </ul>
        </>
      ) : (
        rules.length === 0 &&
        report.rules_evaluated > 0 && (
          // The report names only the rules that failed, so without the pack the ones that ran
          // cleanly cannot be listed. Said rather than left as a shorter list.
          <p className="qtrace-note">
            The vocabulary could not be loaded, so the rules that ran cleanly cannot be named
            here, only the {unable.size} that could not check. The counts above are the
            server's and are unaffected.
          </p>
        )
      )}

      {report.inferences.length > 0 && (
        <>
          <div className="qtrace-sublabel" style={{ marginTop: 16 }}>Staged for review</div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Conclusion</th>
                <th>Rule</th>
                <th>Matter</th>
                <th className="num">Premises</th>
                <th className="num">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {report.inferences.map((i) => (
                <tr key={i.assertion_id}>
                  <td>
                    <strong>{i.subject_id}</strong> <span className="prov-pred">{i.predicate}</span>{' '}
                    <strong>{i.object_id}</strong>
                  </td>
                  <td>
                    <code style={{ fontSize: 11 }}>{i.rule_id}</code>
                  </td>
                  <td className="dim">{i.matter_id ?? '-'}</td>
                  <td className="num">{i.premises.length}</td>
                  <td className="num">{i.confidence.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </>
  )
}

interface RuleRow {
  ruleId: string
  key?: string
  label: string
  detail: string
  rule?: RuleDef
}

/** One of the three failure states. Same shape each time, so the difference is the words. */
function RuleGroup({
  title,
  tag,
  note,
  rows,
  tone = 'red',
}: {
  title: string
  tag: ReactNode
  note: string
  rows: RuleRow[]
  tone?: 'orange' | 'red'
}) {
  // `withheld-block` is red by construction. Recoloured through the same colour-mix rather than
  // with a second copy of the block: a missing fact and a broken rule must not look identical,
  // and neither warrants a new visual language.
  const hue: CSSProperties =
    tone === 'red'
      ? {}
      : {
          borderColor: 'color-mix(in srgb, var(--orange) 34%, transparent)',
          background: 'color-mix(in srgb, var(--orange) 6%, transparent)',
        }
  const item: CSSProperties =
    tone === 'red'
      ? {}
      : {
          borderColor: 'color-mix(in srgb, var(--orange) 26%, transparent)',
          borderLeftColor: 'var(--orange)',
        }

  return (
    <div className="withheld-block" style={{ marginTop: 14, marginBottom: 0, ...hue }}>
      <div className="withheld-block-head">
        <h3>{title}</h3>
        {tag}
      </div>
      <p className="withheld-block-note">{note}</p>
      <div className="withheld-list">
        {rows.map((row) => (
          <div className="withheld-item" key={row.key ?? row.ruleId} style={item}>
            <div className="withheld-item-head">
              <strong>{row.ruleId}</strong>
              {row.rule && <code>version {row.rule.version}</code>}
            </div>
            {row.rule && (
              <div className="withheld-field">
                <span className="withheld-field-label">What it checks</span>
                {row.rule.description}
              </div>
            )}
            <div className="withheld-field">
              <span className="withheld-field-label">{row.label}</span>
              {/* Verbatim. The server names the premise and its endpoint kinds -- "no ADVERSE_TO
                  fact matches (m:Matter)->(p:Party)" is the actionable sentence, and rewording it
                  into prose is what made this diagnostic unusable when it was only in a log. */}
              <code>{row.detail}</code>
            </div>
            {row.rule && (
              <div className="withheld-field">
                <span className="withheld-field-label">Premises it needs</span>
                {row.rule.when.map((w) => (
                  <div key={w}>
                    <code style={{ fontSize: 11 }}>{w}</code>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
