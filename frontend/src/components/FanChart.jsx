import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'

const COLORS = {
  hist: '#666',
  p10: '#6ea8fe',
  p50: '#f5c518',
  p90: '#f59e0b',
  band90: 'rgba(245,197,24,0.08)',
  band50: 'rgba(245,197,24,0.18)',
  fwd: '#4caf7d',
}

export default function FanChart({ data }) {
  const fan = data?.fan
  const history = data?.history
  const forwardPrice = data?.forward_price

  if (!fan?.p50?.length) {
    return (
      <div className="empty-state">
        <span className="empty-state-icon">📈</span>
        <span>Fan chart data unavailable</span>
      </div>
    )
  }

  const histDates = history?.dates ?? []
  const histPrices = history?.prices ?? []

  const histPoints = histDates.map((d, i) => ({
    label: String(d).slice(5),
    price: Number(histPrices[i]),
    type: 'hist',
  }))

  const lastHist = histPoints.length
    ? histPoints[histPoints.length - 1]
    : null

  const joinPoint = lastHist
    ? {
        label: 'Now',
        price: lastHist.price,
        p10: lastHist.price,
        p25: lastHist.price,
        p50: lastHist.price,
        p75: lastHist.price,
        p90: lastHist.price,
        p10Base: lastHist.price,
        p90Band: 0,
        p25Base: lastHist.price,
        p75Band: 0,
      }
    : null

  const fcPoints = fan.p50.map((p50, i) => {
    const p10 = Number(fan.p10?.[i])
    const p25 = Number(fan.p25?.[i])
    const p75 = Number(fan.p75?.[i])
    const p90 = Number(fan.p90?.[i])

    return {
      label: `M${i + 1}`,
      p10,
      p25,
      p50: Number(p50),
      p75,
      p90,
      p10Base: p10,
      p90Band: p90 - p10,
      p25Base: p25,
      p75Band: p75 - p25,
    }
  })

  const chartData = [
    ...histPoints.slice(-12),
    ...(joinPoint ? [joinPoint] : []),
    ...fcPoints,
  ]

  const fmt = value => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return '—'
    }

    return `$${Number(value).toFixed(1)}`
  }

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null

    const visiblePayload = payload.filter(
      p =>
        p.value !== null &&
        p.value !== undefined &&
        !['p10Base', 'p90Band', 'p25Base', 'p75Band'].includes(p.dataKey)
    )

    return (
      <div
        style={{
          background: '#1a1a1a',
          border: '1px solid #2a2a2a',
          borderRadius: 6,
          padding: '8px 12px',
          fontSize: 11,
        }}
      >
        <div style={{ color: '#aaa', marginBottom: 4 }}>
          {label}
        </div>

        {visiblePayload.map(p => (
          <div key={p.name} style={{ color: p.color || '#f0f0f0' }}>
            {p.name}: <strong>{fmt(p.value)}</strong>
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="chart-inner">
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart
          data={chartData}
          margin={{ top: 8, right: 10, left: -10, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />

          <XAxis
            dataKey="label"
            tick={{ fill: '#666', fontSize: 10 }}
            tickLine={false}
          />

          <YAxis
            tick={{ fill: '#666', fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={v => `$${Number(v).toFixed(0)}`}
            width={42}
          />

          <Tooltip content={<CustomTooltip />} />

          <Area
            dataKey="p10Base"
            stackId="outer"
            stroke="none"
            fill="transparent"
            legendType="none"
            isAnimationActive={false}
          />

          <Area
            dataKey="p90Band"
            stackId="outer"
            stroke="none"
            fill={COLORS.band90}
            fillOpacity={1}
            legendType="none"
            isAnimationActive={false}
          />

          <Area
            dataKey="p25Base"
            stackId="inner"
            stroke="none"
            fill="transparent"
            legendType="none"
            isAnimationActive={false}
          />

          <Area
            dataKey="p75Band"
            stackId="inner"
            stroke="none"
            fill={COLORS.band50}
            fillOpacity={1}
            legendType="none"
            isAnimationActive={false}
          />

          <Line
            dataKey="price"
            stroke={COLORS.hist}
            dot={false}
            strokeWidth={1.5}
            name="Historical"
            connectNulls
          />

          <Line
            dataKey="p10"
            stroke={COLORS.p10}
            dot={false}
            strokeWidth={1}
            name="P10"
            strokeDasharray="3 3"
            connectNulls
          />

          <Line
            dataKey="p50"
            stroke={COLORS.p50}
            dot={false}
            strokeWidth={2}
            name="P50 Forecast"
            strokeDasharray="5 3"
            connectNulls
          />

          <Line
            dataKey="p90"
            stroke={COLORS.p90}
            dot={false}
            strokeWidth={1}
            name="P90"
            strokeDasharray="3 3"
            connectNulls
          />

          {forwardPrice !== undefined && forwardPrice !== null && (
            <ReferenceLine
              y={Number(forwardPrice)}
              stroke={COLORS.fwd}
              strokeDasharray="4 3"
              label={{
                value: `Fwd $${Number(forwardPrice).toFixed(0)}`,
                fill: COLORS.fwd,
                fontSize: 10,
                position: 'insideTopRight',
              }}
            />
          )}

          <Legend
            wrapperStyle={{
              fontSize: 10,
              color: '#aaa',
              paddingTop: 4,
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}