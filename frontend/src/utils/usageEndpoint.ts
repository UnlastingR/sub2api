// Keep complete endpoints in the API and database for diagnostics. Collapse
// Gemini model-specific actions only in usage displays and distributions.
export const normalizeUsageEndpoint = (endpoint?: string | null): string => {
  const path = endpoint?.trim() || ''
  return /^\/v1beta\/models\/[^/?#]+/.test(path) ? '/v1beta/models' : path
}

export const isGeminiBatchUsageEndpoint = (endpoint?: string | null): boolean =>
  /^\/v1beta\/models\/[^/?#]+:(batchGenerateContent|asyncBatchEmbedContent|asyncBatchEmbedContents)$/.test(
    endpoint?.trim() || ''
  )
