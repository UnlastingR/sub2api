const GEMINI_LIVE_ENDPOINT = '/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent'
const GEMINI_LIVE_CONSTRAINED_ENDPOINT = '/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained'
const GEMINI_LIVE_ALIAS_ENDPOINT = '/v1beta/live'

// Keep complete endpoints in the API and database for diagnostics. Collapse
// Gemini model-specific actions and long Live gRPC WebSocket paths only in
// usage displays and distributions.
export const normalizeUsageEndpoint = (endpoint?: string | null): string => {
  const path = endpoint?.trim() || ''
  if (path === GEMINI_LIVE_ENDPOINT) return '/ws/gemini-live'
  if (path === GEMINI_LIVE_CONSTRAINED_ENDPOINT) return '/ws/gemini-live-constrained'
  return /^\/v1beta\/models\/[^/?#]+/.test(path) ? '/v1beta/models' : path
}

export const isGeminiLiveUsageEndpoint = (endpoint?: string | null): boolean => {
  const path = endpoint?.trim() || ''
  return path === GEMINI_LIVE_ENDPOINT || path === GEMINI_LIVE_CONSTRAINED_ENDPOINT || path === GEMINI_LIVE_ALIAS_ENDPOINT
}

export const isGeminiBatchUsageEndpoint = (endpoint?: string | null): boolean =>
  /^\/v1beta\/models\/[^/?#]+:(batchGenerateContent|asyncBatchEmbedContent|asyncBatchEmbedContents)$/.test(
    endpoint?.trim() || ''
  )
