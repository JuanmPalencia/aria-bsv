import { useState } from 'react'
import { hashText, hashObject, buildAuditRecord } from '../lib/crypto'

export default function HashDemo() {
  const [prompt, setPrompt] = useState('')
  const [output, setOutput] = useState('')
  const [confidence, setConfidence] = useState('0.92')
  const [modelId, setModelId] = useState('gpt-4o')
  const [systemId, setSystemId] = useState('demo-system')
  const [result, setResult] = useState<object | null>(null)
  const [loading, setLoading] = useState(false)

  async function run() {
    if (!prompt.trim()) return
    setLoading(true)
    try {
      const inputHash = await hashText(prompt)
      const outputHash = await hashText(output || '(no output)')
      const canonObj = { input: prompt, model: modelId }
      const canonHash = await hashObject(canonObj)
      const record = buildAuditRecord({
        systemId,
        modelId,
        inputText: prompt,
        outputText: output || '(no output)',
        confidence: parseFloat(confidence) || 0.9,
        inputHash,
        outputHash,
      })
      setResult({ ...record, canonical_json_hash: canonHash })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="card">
        <h2>Prompt → AuditRecord</h2>
        <p className="desc">
          Enter a prompt and (optionally) a model output. ARIA computes SHA-256 hashes
          of both using canonical JSON serialization — the same deterministic algorithm
          used by the <code>aria-bsv</code> Python SDK and BRC-121.
        </p>

        <div className="row" style={{ gap: '1rem', flexWrap: 'wrap' }}>
          <div className="grow">
            <span className="label">System ID</span>
            <input
              type="text"
              value={systemId}
              onChange={e => setSystemId(e.target.value)}
              placeholder="my-ai-system"
            />
          </div>
          <div className="grow">
            <span className="label">Model ID</span>
            <input
              type="text"
              value={modelId}
              onChange={e => setModelId(e.target.value)}
              placeholder="gpt-4o"
            />
          </div>
          <div style={{ width: '120px' }}>
            <span className="label">Confidence</span>
            <input
              type="text"
              value={confidence}
              onChange={e => setConfidence(e.target.value)}
              placeholder="0.92"
            />
          </div>
        </div>

        <span className="label">Prompt / Input</span>
        <textarea
          rows={4}
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder="What is the capital of France?"
        />

        <span className="label">Model Output (optional)</span>
        <textarea
          rows={3}
          value={output}
          onChange={e => setOutput(e.target.value)}
          placeholder="The capital of France is Paris."
        />

        <button className="btn btn-primary" onClick={run} disabled={loading || !prompt.trim()}>
          {loading ? <span className="spinner" /> : '⚡'} Compute AuditRecord
        </button>
      </div>

      {result && (
        <div className="card">
          <h2>AuditRecord (off-chain preview)</h2>
          <div className="result-box">
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>
          <p style={{ color: 'var(--muted)', fontSize: '0.82rem', marginTop: '0.75rem' }}>
            In production, <code>input_hash</code> and <code>output_hash</code> are committed
            into a Merkle tree and anchored on BSV via BRC-121 EPOCH_OPEN / EPOCH_CLOSE transactions.
          </p>
        </div>
      )}
    </div>
  )
}
