import { describe, expect, it } from 'vitest'
import { isGeminiBatchUsageEndpoint, normalizeUsageEndpoint } from '../usageEndpoint'
import { requestTypeToLegacyStream, resolveUsageRequestType } from '../usageRequestType'

describe('Gemini usage endpoint presentation', () => {
  it.each([
    '/v1beta/models/gemini-2.5-flash-lite:batchGenerateContent',
    '/v1beta/models/gemini-embedding-001:asyncBatchEmbedContent',
    '/v1beta/models/gemini-2.5-flash-lite:streamGenerateContent',
    '/v1beta/models/gemini-2.5-flash-lite:generateContent',
    '/v1beta/models/gemini-2.5-flash-lite/operations/abcdef',
  ])('collapses model-specific endpoint %s for display', (endpoint) => {
    expect(normalizeUsageEndpoint(endpoint)).toBe('/v1beta/models')
  })

  it.each(['/v1beta/models', '/v1beta/decisions', '/v1/embeddings', '/v1beta/batches'])(
    'preserves non-model-action endpoint %s',
    (endpoint) => {
      expect(normalizeUsageEndpoint(endpoint)).toBe(endpoint)
    }
  )

  it('keeps existing sync Batch logs visible as Batch without changing billing mode', () => {
    const endpoint = '/v1beta/models/gemini-2.5-flash-lite:batchGenerateContent'
    expect(isGeminiBatchUsageEndpoint(endpoint)).toBe(true)
    expect(resolveUsageRequestType({ request_type: 'sync', inbound_endpoint: endpoint })).toBe('batch')
    expect(resolveUsageRequestType({ request_type: 'batch', inbound_endpoint: endpoint })).toBe('batch')
    expect(resolveUsageRequestType({ request_type: null, inbound_endpoint: endpoint })).toBe('batch')
    expect(resolveUsageRequestType({ request_type: 'cyber', inbound_endpoint: endpoint })).toBe('cyber')
    expect(resolveUsageRequestType({ request_type: 'stream', inbound_endpoint: endpoint })).toBe('stream')
    expect(resolveUsageRequestType({ request_type: 'sync', inbound_endpoint: '/v1beta/models' })).toBe('sync')
    expect(requestTypeToLegacyStream('batch')).toBeNull()
  })

  it('only classifies Batch actions as Batch', () => {
    expect(isGeminiBatchUsageEndpoint('/v1beta/models/gemini-2.5-flash-lite:generateContent')).toBe(false)
    expect(isGeminiBatchUsageEndpoint('/v1beta/models/gemini-2.5-flash-lite:asyncBatchEmbedContents')).toBe(true)
    expect(isGeminiBatchUsageEndpoint('/v1beta/batches/123')).toBe(false)
  })
})
