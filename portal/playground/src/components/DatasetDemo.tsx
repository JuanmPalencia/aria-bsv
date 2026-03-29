import { useState, useRef, useCallback } from 'react'
import { hashBytes, hashText } from '../lib/crypto'

interface AnchorResult {
  fileName: string
  fileSize: number
  mimeType: string
  contentHash: string
  schemaHash: string | null
  nonce: string
  timestamp: string
  payload: Record<string, unknown>
}

function guessMime(name: string): string {
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  const map: Record<string, string> = {
    csv: 'text/csv', json: 'application/json', txt: 'text/plain',
    png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg',
    parquet: 'application/octet-stream', pkl: 'application/octet-stream',
    pt: 'application/octet-stream', bin: 'application/octet-stream',
  }
  return map[ext] ?? 'application/octet-stream'
}

function randomNonce(): string {
  const arr = new Uint8Array(8)
  crypto.getRandomValues(arr)
  return Array.from(arr).map(b => b.toString(16).padStart(2, '0')).join('')
}

async function anchorFile(file: File): Promise<AnchorResult> {
  const arrayBuffer = await file.arrayBuffer()
  const contentHash = await hashBytes(arrayBuffer)
  const nonce = randomNonce()
  const timestamp = new Date().toISOString()
  const mime = guessMime(file.name)

  // Schema hash: hash of filename + mime type (deterministic proxy for schema)
  const schemaHash = await hashText(`${file.name}|${mime}`)

  const payload = {
    protocol: 'BRC-121',
    type: 'DATASET_ANCHOR',
    content_hash: `sha256:${contentHash}`,
    schema_hash: schemaHash,
    mime_type: mime,
    file_name: file.name,
    file_size: file.size,
    nonce,
    timestamp,
  }

  return {
    fileName: file.name,
    fileSize: file.size,
    mimeType: mime,
    contentHash: `sha256:${contentHash}`,
    schemaHash,
    nonce,
    timestamp,
    payload,
  }
}

export default function DatasetDemo() {
  const [dragging, setDragging] = useState(false)
  const [result, setResult] = useState<AnchorResult | null>(null)
  const [loading, setLoading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const processFile = useCallback(async (file: File) => {
    setLoading(true)
    setResult(null)
    try {
      const r = await anchorFile(file)
      setResult(r)
    } finally {
      setLoading(false)
    }
  }, [])

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      setDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) processFile(file)
    },
    [processFile],
  )

  const onFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) processFile(file)
    },
    [processFile],
  )

  function formatBytes(n: number) {
    if (n < 1024) return `${n} B`
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
    return `${(n / 1024 / 1024).toFixed(2)} MB`
  }

  return (
    <div>
      <div className="card">
        <h2>Dataset Anchoring</h2>
        <p className="desc">
          Drop any file — a CSV, JSON, model weights, or image — to compute its
          SHA-256 content hash and a BRC-121 <code>DATASET_ANCHOR</code> payload.
          In production this payload is broadcast to BSV mainnet as an immutable
          audit record, proving the exact dataset or model version used at inference time.
        </p>

        <div
          className={`drop-zone ${dragging ? 'active' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            style={{ display: 'none' }}
            onChange={onFileChange}
          />
          {loading ? (
            <><span className="spinner" style={{ marginRight: '0.5rem' }} /> Hashing…</>
          ) : (
            <>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📂</div>
              <strong>Drop a file here</strong> or click to browse
              <div style={{ fontSize: '0.82rem', marginTop: '0.3rem' }}>
                CSV, JSON, Parquet, .pt, .pkl, images — any file type
              </div>
            </>
          )}
        </div>
      </div>

      {result && (
        <div className="card">
          <h2>Anchor Payload</h2>

          <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
            <span className="badge badge-ok">✓ Hashed</span>
            <span className="badge badge-info">{result.mimeType}</span>
            <span className="badge badge-info">{formatBytes(result.fileSize)}</span>
          </div>

          <span className="label">Content Hash (SHA-256)</span>
          <div className="result-box">
            <pre>{result.contentHash}</pre>
          </div>

          <span className="label">BRC-121 DATASET_ANCHOR payload</span>
          <div className="result-box">
            <pre>{JSON.stringify(result.payload, null, 2)}</pre>
          </div>

          <p style={{ color: 'var(--muted)', fontSize: '0.82rem', marginTop: '0.75rem' }}>
            The <code>content_hash</code> uniquely identifies this exact file version.
            Changing even one byte produces a completely different hash — any tampering is detectable.
          </p>
        </div>
      )}
    </div>
  )
}
