import { useState, useCallback, useRef } from 'react'
import axios from 'axios'
import './App.css'
import FanChart from './components/FanChart'
import CostHistogram from './components/CostHistogram'
import HedgeSchedule from './components/HedgeSchedule'
import StressChart from './components/StressChart'
import CandidateTable from './components/CandidateTable'

const API = 'http://localhost:8000'

const DEFAULT_PARAMS = {
  forward_price: 80,
  barrels_per_period: 100000,
  horizon: 6,
  max_hedge: 1.0,
  cvar_weight: 1.0,
  model: 'normal',
  n_paths: 2000,
}

function StatPill({ label, value, tone = '' }) {
  return (
    <div className="stat-pill">
      <span className="stat-pill-label">{label}</span>
      <span className={`stat-pill-value ${tone}`}>{value}</span>
    </div>
  )
}

function Slider({ label, value, min, max, step = 1, fmt, onChange }) {
  return (
    <div className="control-item">
      <div className="control-label">
        <span>{label}</span>
        <span className="control-value">{fmt ? fmt(value) : value}</span>
      </div>
      <input type="range" min={min} max={max} step={step}
        value={value} onChange={e => onChange(Number(e.target.value))} />
    </div>
  )
}

export default function App() {
  const [params, setParams] = useState(DEFAULT_PARAMS)
  const [rec, setRec] = useState(null)
  const [stress, setStress] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const debounceRef = useRef(null)

  const setParam = useCallback((key, val) => {
    setParams(prev => ({ ...prev, [key]: val }))
  }, [])

  const runAll = useCallback(async (p) => {
    setLoading(true)
    setError(null)
    try {
      const [recRes, stressRes] = await Promise.all([
        axios.post(`${API}/recommend`, p),
        axios.post(`${API}/stress-test`, {
          forward_price: p.forward_price,
          barrels_per_period: p.barrels_per_period,
          horizon: p.horizon,
          hedge_fraction: p.max_hedge * 0.7,
          custom_shock_pct: 40,
        }),
      ])
      setRec(recRes.data)
      setStress(stressRes.data)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const handleSliderChange = useCallback((key, val) => {
    const next = { ...params, [key]: val }
    setParams(next)
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => runAll(next), 400)
  }, [params, runAll])

  const fmt$ = v => `$${v.toLocaleString()}`
  const fmtPct = v => `${Math.round(v * 100)}%`
  const fmtK = v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : String(v)

  const hedgeTag = rec ? `${rec.hedge_fraction}%` : '—'
  const cvarTag  = rec ? `$${rec.cost.cvar}M` : '—'
  const meanTag  = rec ? `$${rec.cost.mean}M` : '—'

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <span className="header-logo">⬡ CRUDE HEDGE</span>
        <div className="header-divider" />
        {error
          ? <span style={{ color: 'var(--red)', fontSize: 12 }}>⚠ {error}</span>
          : <span className="header-rationale">{rec?.rationale ?? 'Run the model to generate a recommendation.'}</span>
        }
        <div className="header-meta">
          {loading && <span className="spinner" />}
          <StatPill label="Hedge" value={hedgeTag} tone="yellow" />
          <StatPill label="E[Cost]" value={meanTag} />
          <StatPill label="CVaR 95" value={cvarTag} tone="red" />
          <span className={`tag ${rec ? 'tag-dim' : 'tag-dim'}`} style={{ fontSize: 10 }}>
            {rec?.model_name ?? 'No run'}
          </span>
        </div>
      </header>

      <div className="main-layout">
        {/* ── Charts area ── */}
        <main className="charts-area">
          <div className="charts-row">
            <div className="card chart-card">
              <div className="card-title">Price Forecast Fan</div>
              <div className="chart-wrapper">
                {loading && <div className="loading-overlay"><span className="spinner" /></div>}
                {rec ? <FanChart data={rec} /> : <EmptyState icon="📈" text="Run model to see forecast" />}
              </div>
            </div>
            <div className="card chart-card">
              <div className="card-title">Cost Distribution</div>
              <div className="chart-wrapper">
                {loading && <div className="loading-overlay"><span className="spinner" /></div>}
                {rec ? <CostHistogram data={rec.cost_histogram} /> : <EmptyState icon="📊" text="Run model to see cost distribution" />}
              </div>
            </div>
          </div>

          <div className="charts-row">
            <div className="card chart-card">
              <div className="card-title">Hedge Schedule by Period</div>
              <div className="chart-wrapper">
                {loading && <div className="loading-overlay"><span className="spinner" /></div>}
                {rec ? <HedgeSchedule data={rec} /> : <EmptyState icon="📅" text="Run model to see hedge schedule" />}
              </div>
            </div>
            <div className="card chart-card">
              <div className="card-title">Stress Test — Scenario Analysis</div>
              <div className="chart-wrapper">
                {loading && <div className="loading-overlay"><span className="spinner" /></div>}
                {stress ? <StressChart data={stress} /> : <EmptyState icon="⚡" text="Run model to see scenario analysis" />}
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-title">Strategy Candidates Evaluated</div>
            {loading && <div style={{ textAlign: 'center', padding: 16 }}><span className="spinner" /></div>}
            {rec ? <CandidateTable candidates={rec.candidates} /> : <EmptyState icon="🔍" text="Run model to see all evaluated strategies" />}
          </div>
        </main>

        {/* ── Sidebar ── */}
        <aside className="sidebar">
          <div className="sidebar-section-title">Market Inputs</div>

          <div className="control-item">
            <Slider label="Forward Price" value={params.forward_price}
              min={30} max={160} step={1}
              fmt={v => `$${v}/bbl`}
              onChange={v => handleSliderChange('forward_price', v)} />
          </div>
          <div className="control-item">
            <Slider label="Exposure (bbl/mo)" value={params.barrels_per_period}
              min={10000} max={500000} step={10000}
              fmt={v => fmtK(v) + ' bbl'}
              onChange={v => handleSliderChange('barrels_per_period', v)} />
          </div>
          <div className="control-item">
            <Slider label="Horizon (months)" value={params.horizon}
              min={1} max={12} step={1}
              fmt={v => `${v} mo`}
              onChange={v => handleSliderChange('horizon', v)} />
          </div>

          <div className="sidebar-section-title" style={{ marginTop: 4 }}>Risk Controls</div>

          <div className="control-item">
            <Slider label="Max Hedge %" value={params.max_hedge}
              min={0} max={1} step={0.05}
              fmt={fmtPct}
              onChange={v => handleSliderChange('max_hedge', v)} />
          </div>
          <div className="control-item">
            <Slider label="CVaR Weight" value={params.cvar_weight}
              min={0} max={5} step={0.25}
              fmt={v => `${v}×`}
              onChange={v => handleSliderChange('cvar_weight', v)} />
          </div>

          <div className="sidebar-section-title" style={{ marginTop: 4 }}>Model</div>

          <div className="model-toggle">
            {['normal', 'student-t', 'hmm', 'xgb-garch-t'].map(m => (
              <div key={m}
                className={`model-option ${params.model === m ? 'active' : ''}`}
                onClick={() => handleSliderChange('model', m)}>
                <div className="model-dot" />
                {{ normal: 'GBM Normal', 'student-t': 'GBM Student-t', hmm: 'GBM + HMM Regime', 'xgb-garch-t': 'XGB + GARCH-t' }[m]}
              </div>
            ))}
          </div>

          <div className="sidebar-footer">
            <div className="run-config-box">
              paths: <span>{params.n_paths.toLocaleString()}</span><br />
              freq: <span>Monthly</span><br />
              seed: <span>42</span><br />
              model: <span>{rec?.model_name ?? '—'}</span>
            </div>
            <button className="btn-primary" onClick={() => runAll(params)} disabled={loading}>
              {loading ? '⏳ Running…' : '▶  Run Model'}
            </button>
          </div>
        </aside>
      </div>
    </div>
  )
}

function EmptyState({ icon, text }) {
  return (
    <div className="empty-state">
      <span className="empty-state-icon">{icon}</span>
      <span>{text}</span>
    </div>
  )
}
