import {
  ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'

export default function HedgeSchedule({ data }) {
  const { policy_schedule, forward_price } = data
  const horizon = policy_schedule.length

  const chartData = policy_schedule.map((frac, i) => {
    const hedgedBbl = Math.round(frac * (data.cost?.mean > 0 ? 100000 : 100000))
    return {
      period: `M${i + 1}`,
      hedged_pct: Math.round(frac * 100),
      unhedged_pct: Math.round((1 - frac) * 100),
    }
  })

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    return (
      <div style={{ background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: 6, padding: '8px 12px', fontSize: 11 }}>
        <div style={{ color: '#aaa', marginBottom: 4 }}>{label}</div>
        {payload.map(p => (
          <div key={p.name} style={{ color: p.fill || p.stroke || '#f0f0f0' }}>
            {p.name}: <strong>{p.value}{p.unit}</strong>
          </div>
        ))}
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height="100%" minHeight={180}>
      <ComposedChart data={chartData} margin={{ top: 8, right: 10, left: -10, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" vertical={false} />
        <XAxis dataKey="period" tick={{ fill: '#666', fontSize: 10 }} tickLine={false} />
        <YAxis yAxisId="pct" tick={{ fill: '#666', fontSize: 10 }} tickLine={false} axisLine={false}
          tickFormatter={v => `${v}%`} domain={[0, 100]} width={36} />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 10, color: '#aaa', paddingTop: 4 }} />

        <Bar yAxisId="pct" dataKey="hedged_pct" name="Hedged" stackId="vol"
          fill="rgba(245,197,24,0.75)" unit="%" radius={[0,0,0,0]} />
        <Bar yAxisId="pct" dataKey="unhedged_pct" name="Unhedged" stackId="vol"
          fill="rgba(255,255,255,0.08)" unit="%" />

        <Line yAxisId="pct" dataKey="hedged_pct" stroke="#f5c518"
          dot={{ fill: '#f5c518', r: 3 }} strokeWidth={2} name="Hedge %" legendType="none" unit="%" />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
