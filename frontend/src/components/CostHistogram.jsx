import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Legend,
  ResponsiveContainer,
} from 'recharts'

export default function CostHistogram({ data }) {
  const strategy = data?.strategy
  const noHedge = data?.no_hedge
  const cvarLine = data?.cvar_line

  if (!strategy?.counts?.length || !strategy?.edges?.length) {
    return (
      <div className="empty-state">
        <span className="empty-state-icon">📊</span>
        <span>Cost histogram data unavailable</span>
      </div>
    )
  }

  const chartData = strategy.edges.slice(0, -1).map((edge, i) => ({
    bin: Number(edge).toFixed(1),
    strategy: Number(strategy.counts?.[i] ?? 0),
    no_hedge: Number(noHedge?.counts?.[i] ?? 0),
  }))

  const hasNoHedge = Boolean(noHedge?.counts?.length)

  const cvarX =
    cvarLine !== undefined && cvarLine !== null
      ? findNearestBinLabel(chartData, Number(cvarLine))
      : null

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null

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
          ${label}M
        </div>

        {payload.map(p => (
          <div key={p.name} style={{ color: p.color || p.fill }}>
            {p.name}: <strong>{p.value} paths</strong>
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="chart-inner">
      <ResponsiveContainer width="100%" height={280}>
        <BarChart
          data={chartData}
          margin={{ top: 18, right: 10, left: -10, bottom: 0 }}
          barCategoryGap="0%"
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#1e1e1e"
            vertical={false}
          />

          <XAxis
            dataKey="bin"
            tick={{ fill: '#666', fontSize: 9 }}
            tickLine={false}
            interval={7}
            tickFormatter={v => `$${parseFloat(v).toFixed(0)}M`}
          />

          <YAxis
            tick={{ fill: '#666', fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            width={30}
          />

          <Tooltip content={<CustomTooltip />} />

          <Legend
            wrapperStyle={{
              fontSize: 10,
              color: '#aaa',
              paddingTop: 4,
            }}
          />

          {hasNoHedge && (
            <Bar
              dataKey="no_hedge"
              name="No Hedge"
              fill="rgba(224,92,92,0.45)"
            />
          )}

          <Bar
            dataKey="strategy"
            name="Strategy"
            fill="rgba(76,175,125,0.65)"
          />

          {cvarX && (
            <ReferenceLine
              x={cvarX}
              stroke="#e05c5c"
              strokeDasharray="4 3"
              label={{
                value: `CVaR $${Number(cvarLine).toFixed(2)}M`,
                fill: '#e05c5c',
                fontSize: 10,
                position: 'top',
              }}
            />
          )}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function findNearestBinLabel(chartData, value) {
  if (!chartData?.length) return null

  let best = chartData[0]
  let bestDiff = Math.abs(Number(best.bin) - value)

  for (const row of chartData) {
    const diff = Math.abs(Number(row.bin) - value)

    if (diff < bestDiff) {
      best = row
      bestDiff = diff
    }
  }

  return best.bin
}