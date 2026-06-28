import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from 'recharts'

const STRATEGY_COLORS = {
  staggered:  '#6ea8fe',
  trigger:    '#a78bfa',
  volatility: '#34d399',
  hybrid:     '#f59e0b',
  cvar_lp:    '#f472b6',
}

const STRATEGY_LABELS = {
  staggered:  'Staggered',
  trigger:    'Trigger',
  volatility: 'Volatility',
  hybrid:     'Hybrid',
  cvar_lp:    'CVaR-LP',
}

function dotColor(d) {
  if (d.accepted) return '#ffffff'
  return STRATEGY_COLORS[d.strategy_type] ?? '#555'
}

function dotOpacity(d) {
  return d.accepted ? 1.0 : 0.45
}

function dotSize(d) {
  return d.accepted ? 90 : 28
}

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null

  return (
    <div
      style={{
        background: '#1a1a1a',
        border: `1px solid ${d.accepted ? '#ffffff44' : '#2a2a2a'}`,
        borderRadius: 6,
        padding: '8px 12px',
        fontSize: 11,
        minWidth: 160,
      }}
    >
      <div style={{ color: '#aaa', marginBottom: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
        <span
          style={{
            display: 'inline-block',
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: dotColor(d),
          }}
        />
        {STRATEGY_LABELS[d.strategy_type] ?? d.strategy_type}
        {d.accepted && (
          <span style={{ color: '#4caf7d', fontWeight: 700, marginLeft: 4 }}>
            ★ Selected
          </span>
        )}
      </div>
      <div style={{ color: '#ccc' }}>
        Hedge: <strong style={{ color: '#f5c518' }}>{d.hedge_pct}%</strong>
      </div>
      <div style={{ color: '#ccc' }}>
        E[Cost]: <strong>${d.x?.toFixed(2)}M</strong>
      </div>
      <div style={{ color: '#ccc' }}>
        CVaR 95: <strong style={{ color: '#e05c5c' }}>${d.y?.toFixed(2)}M</strong>
      </div>
      <div style={{ color: '#ccc' }}>
        Blended score: <strong>{d.blended?.toFixed(2)}</strong>
      </div>
    </div>
  )
}

const LegendDot = ({ color, label }) => (
  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, marginRight: 10, fontSize: 10, color: '#888' }}>
    <span style={{ width: 7, height: 7, borderRadius: '50%', background: color, display: 'inline-block', opacity: 0.7 }} />
    {label}
  </span>
)

export default function RiskRewardChart({ candidates }) {
  if (!candidates?.length) {
    return (
      <div className="empty-state">
        <span className="empty-state-icon">🎯</span>
        <span>Risk-reward data unavailable</span>
      </div>
    )
  }

  const chartData = candidates.map(c => ({
    ...c,
    x: c.expected_cost,
    y: c.cvar,
  }))

  const winner = chartData.find(d => d.accepted)

  const allX = chartData.map(d => d.x).filter(Number.isFinite)
  const allY = chartData.map(d => d.y).filter(Number.isFinite)

  const xPad = (Math.max(...allX) - Math.min(...allX)) * 0.08 || 1
  const yPad = (Math.max(...allY) - Math.min(...allY)) * 0.08 || 1

  const xDomain = [
    Math.max(0, Math.min(...allX) - xPad),
    Math.max(...allX) + xPad,
  ]
  const yDomain = [
    Math.max(0, Math.min(...allY) - yPad),
    Math.max(...allY) + yPad,
  ]

  return (
    <div className="chart-inner">
      {/* mini legend */}
      <div style={{ paddingLeft: 4, paddingBottom: 4, display: 'flex', flexWrap: 'wrap' }}>
        {Object.entries(STRATEGY_LABELS).map(([k, v]) => (
          <LegendDot key={k} color={STRATEGY_COLORS[k]} label={v} />
        ))}
        <LegendDot color="#ffffff" label="Selected" />
      </div>

      <ResponsiveContainer width="100%" height={248}>
        <ScatterChart margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />

          <XAxis
            type="number"
            dataKey="x"
            domain={xDomain}
            tick={{ fill: '#666', fontSize: 10 }}
            tickLine={false}
            tickFormatter={v => `$${v.toFixed(0)}M`}
            label={{
              value: 'Expected Cost →',
              position: 'insideBottomRight',
              offset: -4,
              fill: '#444',
              fontSize: 9,
            }}
          />

          <YAxis
            type="number"
            dataKey="y"
            domain={yDomain}
            tick={{ fill: '#666', fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={v => `$${v.toFixed(0)}M`}
            width={46}
            label={{
              value: 'CVaR 95 →',
              angle: -90,
              position: 'insideLeft',
              offset: 14,
              fill: '#444',
              fontSize: 9,
            }}
          />

          <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#333', strokeWidth: 1 }} />

          {/* "better" direction annotation */}
          <ReferenceLine
            x={xDomain[0] + (xDomain[1] - xDomain[0]) * 0.05}
            stroke="transparent"
            label={{
              value: '← lower is better',
              position: 'insideTopRight',
              fill: '#333',
              fontSize: 9,
            }}
          />

          {/* winner crosshairs */}
          {winner && (
            <>
              <ReferenceLine
                x={winner.x}
                stroke="#ffffff18"
                strokeDasharray="3 3"
              />
              <ReferenceLine
                y={winner.y}
                stroke="#ffffff18"
                strokeDasharray="3 3"
              />
            </>
          )}

          <Scatter data={chartData} isAnimationActive={false}>
            {chartData.map((d, i) => (
              <Cell
                key={i}
                fill={dotColor(d)}
                fillOpacity={dotOpacity(d)}
                r={Math.sqrt(dotSize(d))}
                stroke={d.accepted ? '#4caf7d' : 'transparent'}
                strokeWidth={d.accepted ? 2 : 0}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}
