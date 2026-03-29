import { useState } from 'react'

type VerifyStatus = 'idle' | 'loading' | 'ok' | 'error'

interface TxData {
  txid: string
  outputs: { script: string; value: number }[]
}

interface VerifyResult {
  valid: boolean
  openTxid: string
  closeTxid: string
  openPayload?: Record<string, unknown>
  closePayload?: Record<string, unknown>
  error?: string
  raw?: string
}

const WOC_BASE = 'https://api.whatsonchain.com/v1/bsv/main/tx'

async function fetchTx(txid: string): Promise<TxData> {
  const res = await fetch(`${WOC_BASE}/${txid}`)
  if (!res.ok) throw new Error(`WoC returned ${res.status} for ${txid}`)
  const data = await res.json()
  const outputs = (data.vout ?? []).map((o: { scriptPubKey: { hex: string }; value: number }) => ({
    script: o.scriptPubKey?.hex ?? '',
    value: o.value ?? 0,
  }))
  return { txid, outputs }
}

function extractAriaPayload(scriptHex: string): Record<string, unknown> | null {
  // OP_RETURN scripts start with 6a (after optional 00 prefix)
  const clean = scriptHex.startsWith('00') ? scriptHex.slice(2) : scriptHex
  if (!clean.startsWith('6a')) return null
  try {
    // Strip OP_RETURN (6a) + length byte(s) and decode UTF-8
    const withoutOpReturn = clean.slice(2)
    // Try to find JSON — skip length-prefixed pushdata
    const hex = withoutOpReturn.length > 2 ? withoutOpReturn.slice(2) : withoutOpReturn
    const bytes = Uint8Array.from(hex.match(/.{2}/g)!.map(b => parseInt(b, 16)))
    const text = new TextDecoder('utf-8', { fatal: false }).decode(bytes)
    // Find JSON object in the decoded text
    const start = text.indexOf('{')
    if (start === -1) return { raw: text.trim() }
    return JSON.parse(text.slice(start))
  } catch {
    return null
  }
}

export default function VerifyDemo() {
  const [openTxid, setOpenTxid] = useState('')
  const [closeTxid, setCloseTxid] = useState('')
  const [status, setStatus] = useState<VerifyStatus>('idle')
  const [result, setResult] = useState<VerifyResult | null>(null)

  async function verify() {
    const open = openTxid.trim()
    const close = closeTxid.trim()
    if (!open || !close) return

    setStatus('loading')
    setResult(null)

    try {
      const [openTx, closeTx] = await Promise.all([fetchTx(open), fetchTx(close)])

      // Find OP_RETURN outputs
      const openScript = openTx.outputs.find(o => o.script.includes('6a'))
      const closeScript = closeTx.outputs.find(o => o.script.includes('6a'))

      const openPayload = openScript ? extractAriaPayload(openScript.script) : null
      const closePayload = closeScript ? extractAriaPayload(closeScript.script) : null

      // Basic chain-link validation: close must reference open's txid
      const linkedCorrectly =
        closePayload &&
        typeof closePayload['prev_txid'] === 'string' &&
        (closePayload['prev_txid'] as string).toLowerCase() === open.toLowerCase()

      const valid = !!(openPayload && closePayload && linkedCorrectly)

      setResult({
        valid,
        openTxid: open,
        closeTxid: close,
        openPayload: openPayload ?? undefined,
        closePayload: closePayload ?? undefined,
        raw: undefined,
      })
      setStatus(valid ? 'ok' : 'error')
    } catch (err) {
      setResult({
        valid: false,
        openTxid: open,
        closeTxid: close,
        error: err instanceof Error ? err.message : String(err),
      })
      setStatus('error')
    }
  }

  function loadExample() {
    setOpenTxid('0c2e80c23d17f4231f0ee9cbe05a2dc427ed13a1f0c8caa9cce95901034f4190')
    setCloseTxid('f6020fcb79e9a5489c24e42814363abc669ba628692a8a4f9ba984090f12c9b6')
  }

  return (
    <div>
      <div className="card">
        <h2>Epoch Verifier</h2>
        <p className="desc">
          Enter EPOCH_OPEN and EPOCH_CLOSE transaction IDs from BSV mainnet.
          The verifier fetches the OP_RETURN payloads from WhatsOnChain and checks
          that the <code>prev_txid</code> link is intact — the core BRC-121 chain-of-custody guarantee.
        </p>

        <span className="label">EPOCH_OPEN txid</span>
        <input
          type="text"
          value={openTxid}
          onChange={e => setOpenTxid(e.target.value)}
          placeholder="64-char hex txid"
          className="mono"
        />
        <span className="label">EPOCH_CLOSE txid</span>
        <input
          type="text"
          value={closeTxid}
          onChange={e => setCloseTxid(e.target.value)}
          placeholder="64-char hex txid"
          className="mono"
        />

        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          <button
            className="btn btn-primary"
            onClick={verify}
            disabled={status === 'loading' || !openTxid.trim() || !closeTxid.trim()}
          >
            {status === 'loading' ? <span className="spinner" /> : '🔗'} Verify Epoch
          </button>
          <button
            className="btn"
            style={{ background: 'var(--surface2)', color: 'var(--muted)', marginTop: '0.75rem' }}
            onClick={loadExample}
          >
            Load smoke-test example
          </button>
        </div>
      </div>

      {result && (
        <div className="card">
          <h2>
            Result{' '}
            <span className={`badge ${result.valid ? 'badge-ok' : 'badge-err'}`}>
              {result.valid ? '✓ VALID' : '✗ INVALID'}
            </span>
          </h2>

          {result.error ? (
            <p style={{ color: 'var(--danger)', fontFamily: 'var(--font)', fontSize: '0.88rem' }}>
              {result.error}
            </p>
          ) : (
            <>
              {result.openPayload && (
                <>
                  <span className="label">EPOCH_OPEN payload</span>
                  <div className="result-box">
                    <pre>{JSON.stringify(result.openPayload, null, 2)}</pre>
                  </div>
                </>
              )}
              {result.closePayload && (
                <>
                  <span className="label">EPOCH_CLOSE payload</span>
                  <div className="result-box">
                    <pre>{JSON.stringify(result.closePayload, null, 2)}</pre>
                  </div>
                </>
              )}
              {!result.valid && (
                <p style={{ color: 'var(--danger)', fontSize: '0.88rem', marginTop: '0.75rem' }}>
                  ⚠ The EPOCH_CLOSE <code>prev_txid</code> does not match the EPOCH_OPEN txid.
                  The chain-of-custody link is broken.
                </p>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
