import { describe, expect, it } from 'vitest'
import { isGeminiBatchUsageEndpoint, isGeminiLiveUsageEndpoint, normalizeUsageEndpoint } from '../usageEndpoint'
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

  it.each([
    [
      '/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent',
      '/ws/gemini-live',
    ],
    [
      '/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained',
      '/ws/gemini-live-constrained',
    ],
  ])('shortens Gemini Live WebSocket endpoint %s to %s', (endpoint, want) => {
    expect(isGeminiLiveUsageEndpoint(endpoint)).toBe(true)
    expect(normalizeUsageEndpoint(endpoint)).toBe(want)
  })

  it.each(['/v1beta/models', '/v1beta/decisions', '/v1/embeddings', '/v1beta/batches'])(
    'preserves non-model-action endpoint %s',
    (endpoint) => {
      expect(isGeminiLiveUsageEndpoint(endpoint)).toBe(false)
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

  it('keeps existing stream Gemini Live logs visible as Live while preserving explicit cyber/live', () => {
    const endpoint = '/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent'
    const constrainedEndpoint = '/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained'
    const aliasEndpoint = '/v1beta/live'
    expect(isGeminiLiveUsageEndpoint(aliasEndpoint)).toBe(true)
    expect(normalizeUsageEndpoint(aliasEndpoint)).toBe(aliasEndpoint)
    expect(resolveUsageRequestType({ request_type: 'stream', inbound_endpoint: endpoint })).toBe('live')
    expect(resolveUsageRequestType({ request_type: 'stream', inbound_endpoint: constrainedEndpoint })).toBe('live')
    expect(resolveUsageRequestType({ request_type: 'stream', inbound_endpoint: aliasEndpoint })).toBe('live')
    expect(resolveUsageRequestType({ request_type: null, stream: true, inbound_endpoint: endpoint })).toBe('live')
    expect(resolveUsageRequestType({ request_type: 'live', inbound_endpoint: endpoint })).toBe('live')
    expect(resolveUsageRequestType({ request_type: 'cyber', inbound_endpoint: endpoint })).toBe('cyber')
    expect(resolveUsageRequestType({ request_type: 'stream', inbound_endpoint: '/v1beta/models' })).toBe('stream')
    expect(requestTypeToLegacyStream('live')).toBeNull()
  })

  it('only classifies Batch actions as Batch', () => {
    expect(isGeminiBatchUsageEndpoint('/v1beta/models/gemini-2.5-flash-lite:generateContent')).toBe(false)
    expect(isGeminiBatchUsageEndpoint('/v1beta/models/gemini-2.5-flash-lite:asyncBatchEmbedContents')).toBe(true)
    expect(isGeminiBatchUsageEndpoint('/v1beta/batches/123')).toBe(false)
  })
})
