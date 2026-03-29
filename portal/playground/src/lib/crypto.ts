/**
 * Canonical JSON serialization compatible with ARIA BRC-121 / aria-bsv Python SDK.
 * Keys are sorted lexicographically. Arrays preserve order.
 */
export function canonicalJson(value: unknown): string {
  if (value === null || value === undefined) return 'null'
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'number') {
    if (!isFinite(value)) throw new Error('Non-finite number not allowed in canonical JSON')
    return JSON.stringify(value)
  }
  if (typeof value === 'string') return JSON.stringify(value)
  if (Array.isArray(value)) {
    return '[' + value.map(canonicalJson).join(',') + ']'
  }
  if (typeof value === 'object') {
    const keys = Object.keys(value as object).sort()
    return '{' + keys.map(k => `${JSON.stringify(k)}:${canonicalJson((value as Record<string, unknown>)[k])}`).join(',') + '}'
  }
  throw new Error(`Unsupported type: ${typeof value}`)
}

/** SHA-256 of the canonical JSON encoding of value. Returns lowercase hex. */
export async function hashObject(value: unknown): Promise<string> {
  const json = canonicalJson(value)
  return hashText(json)
}

/** SHA-256 of a UTF-8 string. Returns lowercase hex. */
export async function hashText(text: string): Promise<string> {
  const encoder = new TextEncoder()
  const data = encoder.encode(text)
  const buffer = await crypto.subtle.digest('SHA-256', data)
  return Array.from(new Uint8Array(buffer)).map(b => b.toString(16).padStart(2, '0')).join('')
}

/** SHA-256 of raw bytes. Returns lowercase hex. */
export async function hashBytes(data: ArrayBuffer | Uint8Array): Promise<string> {
  const buf: ArrayBuffer = data instanceof Uint8Array ? data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer : data
  const buffer = await crypto.subtle.digest('SHA-256', buf)
  return Array.from(new Uint8Array(buffer)).map(b => b.toString(16).padStart(2, '0')).join('')
}

/** Build an AuditRecord preview (off-chain, for display) */
export function buildAuditRecord(params: {
  systemId: string
  modelId: string
  inputText: string
  outputText: string
  confidence: number
  inputHash: string
  outputHash: string
}) {
  const now = new Date().toISOString()
  const recordId = `rec_demo_${Date.now()}`
  return {
    record_id: recordId,
    system_id: params.systemId,
    model_id: params.modelId,
    input: params.inputText,
    output: params.outputText,
    confidence: params.confidence,
    input_hash: params.inputHash,
    output_hash: params.outputHash,
    timestamp: now,
    protocol: 'BRC-121',
    note: 'Off-chain preview — not broadcast to BSV mainnet in playground mode',
  }
}
