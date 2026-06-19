import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Cell, LabelList, ResponsiveContainer, Legend,
} from 'recharts'

const SHOCK_COLOR = s => s > 0 ? '#e05c5c' : '#4caf7d'

export default function StressChart({ data }) {
  const { scenarios } = data

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    const row = scenarios.find(s => s.scenario === label)
    const saved = row?.savings ?? 0
    return (
      <div style={{ background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: 6, padding: '8px 12px', fontSize: 11 }}>
        <div style={{ color: '#aaa', marginBottom: 6, fontWeight: 600 }}>{label}</div>
        <div style={{ color: '#e05c5c' }}>No Hedge: <strong>${payload.find(p=>p.dataKey==='no_hedge_cost')?.value}M</strong></div>
        <div style={{ color: '#4caf7d' }}>Hedged:   <strong>${payload.find(p=>p.dataKey==='hedged_cost')?.value}M</strong></div>
        <div style={{ color: saved >= 0 ? '#4caf7d' : '#e05c5c', marginTop: 4, borderTop: '1px solid #2a2a2a', paddingTop: 4 }}>
          Savings: <strong>{saved >= 0 ? '+' : ''}${saved}M</strong>
        </div>
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height="100%" minHeight={180}>
      <BarChart data={scenarios} margin={{ top: 8, right: 10, left: -10, bottom: 0 }} barCategoryGap="30%">
        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" vertical={false} />
        <XAxis dataKey="scenario" tick={{ fill: '#666', fontSize: 9 }} tickLine={false}
          interval={0} angle={-12} textAnchor="end" height={36} />
        <YAxis tick={{ fill: '#666', fontSize: 10 }} tickLine={false} axisLine={false}
          tickFormatter={v => `$${v}M`} width={46} />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 10, color: '#aaa', paddingTop: 4 }} />

        <Bar dataKey="no_hedge_cost" name="No Hedge" fill="rgba(224,92,92,0.55)" radius={[3,3,0,0]}>
          {scenarios.map(s => (
            <Cell key={s.scenario} fill="rgba(224,92,92,0.55)" />
          ))}
        </Bar>
        <Bar dataKey="hedged_cost" name="Hedged" fill="rgba(76,175,125,0.7)" radius={[3,3,0,0]}>
          <LabelList dataKey="savings" position="top"
            formatter={v => v >= 0 ? `+$${v}M` : `-$${Math.abs(v)}M`}
            style={{ fill: '#aaa', fontSize: 9 }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
