import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, Legend, ResponsiveContainer, Cell,
} from 'recharts'

export default function CostHistogram({ data }) {
  const { strategy, no_hedge, cvar_line } = data

  // Build unified bins from strategy histogram
  const chartData = strategy.edges.slice(0, -1).map((edge, i) => ({
    bin: edge.toFixed(1),
    strategy: strategy.counts[i],
    no_hedge: no_hedge.counts[i] ?? 0,
  }))

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    return (
      <div style={{ background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: 6, padding: '8px 12px', fontSize: 11 }}>
        <div style={{ color: '#aaa', marginBottom: 4 }}>${label}M</div>
        {payload.map(p => (
          <div key={p.name} style={{ color: p.fill }}>
            {p.name}: <strong>{p.value} paths</strong>
          </div>
        ))}
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height="100%" minHeight={180}>
      <BarChart data={chartData} margin={{ top: 8, right: 10, left: -10, bottom: 0 }} barCategoryGap="0%">
        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" vertical={false} />
        <XAxis dataKey="bin" tick={{ fill: '#666', fontSize: 9 }} tickLine={false}
          tickFormatter={v => `$${parseFloat(v).toFixed(0)}M`} interval={7} />
        <YAxis tick={{ fill: '#666', fontSize: 10 }} tickLine={false} axisLine={false} width={30} />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 10, color: '#aaa', paddingTop: 4 }} />

        <Bar dataKey="no_hedge" name="No Hedge" fill="rgba(224,92,92,0.45)" />
        <Bar dataKey="strategy" name="Strategy" fill="rgba(76,175,125,0.65)" />

        <ReferenceLine x={cvar_line?.toFixed(1)} stroke="#e05c5c" strokeDasharray="4 3"
          label={{ value: `CVaR $${cvar_line}M`, fill: '#e05c5c', fontSize: 10, position: 'top' }} />
      </BarChart>
    </ResponsiveContainer>
  )
}
