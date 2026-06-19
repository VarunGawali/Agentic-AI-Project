import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ResponsiveContainer,
} from 'recharts'

const COLORS = {
  hist:  '#666',
  p50:   '#f5c518',
  band90: 'rgba(245,197,24,0.08)',
  band50: 'rgba(245,197,24,0.18)',
  fwd:   '#4caf7d',
}

export default function FanChart({ data }) {
  const { history, fan, forward_price, policy_schedule } = data
  const horizon = fan.p50.length

  // Historical points
  const histPoints = history.dates.map((d, i) => ({
    label: d.slice(5),
    price: history.prices[i],
    type: 'hist',
  }))

  // Forecast points (month labels)
  const fcPoints = fan.p50.map((p50, i) => ({
    label: `M${i + 1}`,
    p10: fan.p10[i],
    p25: fan.p25[i],
    p50,
    p75: fan.p75[i],
    p90: fan.p90[i],
    type: 'fc',
  }))

  // Join: last historical + forecast
  const lastHist = histPoints.at(-1)
  const joinPoint = { label: 'Now', price: lastHist.price, p10: lastHist.price, p25: lastHist.price, p50: lastHist.price, p75: lastHist.price, p90: lastHist.price }

  // Only show last 12 hist for readability
  const chartData = [...histPoints.slice(-12), joinPoint, ...fcPoints]

  const fmt = v => `$${v?.toFixed(1)}`

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    return (
      <div style={{ background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: 6, padding: '8px 12px', fontSize: 11 }}>
        <div style={{ color: '#aaa', marginBottom: 4 }}>{label}</div>
        {payload.filter(p => p.value != null).map(p => (
          <div key={p.name} style={{ color: p.color || '#f0f0f0' }}>
            {p.name}: <strong>{fmt(p.value)}</strong>
          </div>
        ))}
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height="100%" minHeight={180}>
      <ComposedChart data={chartData} margin={{ top: 8, right: 10, left: -10, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
        <XAxis dataKey="label" tick={{ fill: '#666', fontSize: 10 }} tickLine={false} />
        <YAxis tick={{ fill: '#666', fontSize: 10 }} tickLine={false} axisLine={false}
          tickFormatter={v => `$${v.toFixed(0)}`} width={42} />
        <Tooltip content={<CustomTooltip />} />

        {/* P10-P90 outer band */}
        <Area dataKey="p10" stroke="none" fill="transparent" legendType="none" />
        <Area dataKey="p90" stroke="none" fill={COLORS.band90}
          fillOpacity={1} legendType="none" baseLine={d => d.p10} />

        {/* P25-P75 inner band */}
        <Area dataKey="p25" stroke="none" fill="transparent" legendType="none" />
        <Area dataKey="p75" stroke="none" fill={COLORS.band50}
          fillOpacity={1} legendType="none" baseLine={d => d.p25} />

        {/* Historical price */}
        <Line dataKey="price" stroke={COLORS.hist} dot={false}
          strokeWidth={1.5} name="Historical" connectNulls />

        {/* P50 median */}
        <Line dataKey="p50" stroke={COLORS.p50} dot={false}
          strokeWidth={2} name="P50 Forecast" strokeDasharray="5 3" connectNulls />

        {/* Forward price reference */}
        <ReferenceLine y={forward_price} stroke={COLORS.fwd} strokeDasharray="4 3"
          label={{ value: `Fwd $${forward_price}`, fill: COLORS.fwd, fontSize: 10, position: 'insideTopRight' }} />

        <Legend wrapperStyle={{ fontSize: 10, color: '#aaa', paddingTop: 4 }} />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
