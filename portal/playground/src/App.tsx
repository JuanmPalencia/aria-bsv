import { useState } from 'react'
import HashDemo from './components/HashDemo'
import VerifyDemo from './components/VerifyDemo'
import DatasetDemo from './components/DatasetDemo'

type Tab = 'hash' | 'verify' | 'dataset'

const TABS: { id: Tab; label: string }[] = [
  { id: 'hash', label: '⚡ Prompt → AuditRecord' },
  { id: 'verify', label: '🔗 Epoch Verifier' },
  { id: 'dataset', label: '📂 Dataset Anchoring' },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('hash')

  return (
    <div className="App">
      <header>
        <h1>ARIA BRC-121 Playground</h1>
        <p>
          Interactive demos for{' '}
          <a href="https://github.com/JuanmPalencia/aria-bsv" target="_blank" rel="noreferrer"
            style={{ color: 'var(--accent2)', textDecoration: 'none' }}>
            aria-bsv
          </a>{' '}
          — Auditable Real-time Inference Architecture
        </p>
      </header>

      <div className="tabs">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`tab-btn ${tab === t.id ? 'active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'hash' && <HashDemo />}
      {tab === 'verify' && <VerifyDemo />}
      {tab === 'dataset' && <DatasetDemo />}

      <footer>
        <p>
          <a href="https://github.com/JuanmPalencia/aria-bsv" target="_blank" rel="noreferrer">GitHub</a>
          {' · '}
          <a href="https://pypi.org/project/aria-bsv/" target="_blank" rel="noreferrer">PyPI</a>
          {' · '}
          <a href="https://github.com/bitcoin-sv/BRCs/pull/129" target="_blank" rel="noreferrer">BRC-121 PR</a>
          {' · '}
          Protocol: BRC-121 · SDK: v0.4.0
        </p>
      </footer>
    </div>
  )
}
