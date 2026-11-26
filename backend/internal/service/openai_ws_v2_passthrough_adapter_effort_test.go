//go:build unit

package service

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestWSPassthroughUsageMeta_InitFromFirstFrame_MappedModelCandidate(t *testing.T) {
	body := []byte(`{"type":"response.create","model":"sol","reasoning":{"effort":"max"}}`)

	meta := newOpenAIWSPassthroughUsageMeta("sol", body)
	meta.initFromFirstFrame(body, "gpt-5.6-sol")

	got := meta.reasoningEffort.Load()
	require.NotNil(t, got, "reasoning effort should be set")
	require.Equal(t, "max", *got, "mapped model gpt-5.6-sol should preserve max")
}

func TestWSPassthroughUsageMeta_InitFromFirstFrame_NonGPT56FallsBackToXHigh(t *testing.T) {
	body := []byte(`{"type":"response.create","model":"gpt-5.4","reasoning":{"effort":"max"}}`)

	meta := newOpenAIWSPassthroughUsageMeta("gpt-5.4", body)
	meta.initFromFirstFrame(body, "gpt-5.4")
	meta.captureRequestedReasoningEffort(body, "gpt-5.4")

	got := meta.reasoningEffort.Load()
	require.NotNil(t, got)
	require.Equal(t, "xhigh", *got, "non-5.6 model should normalize max to xhigh")
	requested := meta.requestedReasoningEffort.Load()
	require.NotNil(t, requested)
	require.Equal(t, "max", *requested, "usage should keep the pre-mapping requested effort")
}

func TestWSPassthroughUsageMeta_UpdateFromResponseCreate_MappedModelCandidate(t *testing.T) {
	body := []byte(`{"type":"response.create","model":"sol","reasoning":{"effort":"max"}}`)

	meta := newOpenAIWSPassthroughUsageMeta("sol", body)
	meta.updateFromResponseCreate(body, "gpt-5.6-sol", "sol")

	got := meta.reasoningEffort.Load()
	require.NotNil(t, got)
	require.Equal(t, "max", *got, "mapped model should preserve max on multi-turn update")
}

func TestWSPassthroughUsageMeta_UsesLatestConfigurationUpdateEffort(t *testing.T) {
	body := []byte(`{"type":"response.create","model":"gpt-6.2-sol","reasoning":{"effort":"medium"},"input":[{"role":"user","content":"inspect this"},{"type":"configuration_update","reasoning":{"effort":"xhigh"}}]}`)

	meta := newOpenAIWSPassthroughUsageMeta("gpt-6.2-sol", body)
	meta.initFromFirstFrame(body, "gpt-6.2-sol")
	meta.captureRequestedReasoningEffort(body, "gpt-6.2-sol")

	forwarded := meta.reasoningEffort.Load()
	require.NotNil(t, forwarded)
	require.Equal(t, "xhigh", *forwarded, "the current turn effort must override the top-level session default")
	requested := meta.requestedReasoningEffort.Load()
	require.NotNil(t, requested)
	require.Equal(t, "xhigh", *requested)

	nextTurn := []byte(`{"type":"response.create","model":"gpt-6.2-sol","reasoning":{"effort":"medium"},"input":[{"type":"configuration_update","reasoning":{"effort":"high"}}]}`)
	meta.updateFromResponseCreate(nextTurn, "gpt-6.2-sol", "gpt-6.2-sol")
	forwarded = meta.reasoningEffort.Load()
	require.NotNil(t, forwarded)
	require.Equal(t, "high", *forwarded, "each WebSocket turn must use its latest configuration update")
}
