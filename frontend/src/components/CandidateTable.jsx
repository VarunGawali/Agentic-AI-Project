export default function CandidateTable({ candidates }) {
  if (!candidates?.length) return null

  const noHedge = candidates.find(c => c.hedge_pct === 0)
  const winner = candidates.find(c => c.accepted)

  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="candidates-table">
        <thead>
          <tr>
            <th>Hedge %</th>
            <th>E[Cost] ($M)</th>
            <th>CVaR 95 ($M)</th>
            <th>Blended Score</th>
            <th>vs No Hedge ($M)</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {candidates.map((c, i) => {
            const delta = noHedge ? (c.expected_cost - noHedge.expected_cost) : null
            return (
              <tr key={i} className={c.accepted ? 'winner' : ''}>
                <td>{c.hedge_pct}%</td>
                <td>{c.expected_cost.toFixed(2)}</td>
                <td>{c.cvar.toFixed(2)}</td>
                <td>{c.blended.toFixed(2)}</td>
                <td>
                  {delta !== null
                    ? <span className={delta <= 0 ? 'savings-pos' : 'savings-neg'}>
                        {delta <= 0 ? '▼' : '▲'} {Math.abs(delta).toFixed(2)}
                      </span>
                    : '—'
                  }
                </td>
                <td>{c.accepted && <span className="badge-winner">✓ CHOSEN</span>}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}