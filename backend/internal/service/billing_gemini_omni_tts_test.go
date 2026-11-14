package service

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestGeminiOmniBilling_VideoAndTextModalitySplit(t *testing.T) {
	svc := NewBillingService(nil, nil)

	p12, err := svc.GetModelPricing("gemini-omni-1.2-flash")
	require.NoError(t, err)
	require.InDelta(t, 1.5e-6, p12.InputPricePerToken, 1e-12)
	require.InDelta(t, 9e-6, p12.OutputPricePerToken, 1e-12)
	require.InDelta(t, 12e-6, p12.VideoOutputPricePerToken, 1e-12)

	// 1,000,000 input + 500,000 text output + 1,000,000 video output (OutputTokens holds total = 1,500,000).
	breakdown := svc.computeTokenBreakdown(p12, UsageTokens{
		InputTokens:       1_000_000,
		OutputTokens:      1_500_000,
		VideoOutputTokens: 1_000_000,
	}, 1.0, "", true)
	require.InDelta(t, 1.5, breakdown.InputCost, 1e-9)
	require.InDelta(t, 0.5*9.0+12.0, breakdown.OutputCost, 1e-9)
	require.InDelta(t, 18.0, breakdown.TotalCost, 1e-9)

	p11, err := svc.GetModelPricing("gemini-omni-1.1-flash")
	require.NoError(t, err)
	require.InDelta(t, 17.5e-6, p11.VideoOutputPricePerToken, 1e-12)
	breakdown11 := svc.computeTokenBreakdown(p11, UsageTokens{
		InputTokens:       100,
		OutputTokens:      1931,
		VideoOutputTokens: 1931,
	}, 1.0, "", true)
	require.InDelta(t, 100*1.5e-6, breakdown11.InputCost, 1e-9)
	require.InDelta(t, 1931*17.5e-6, breakdown11.OutputCost, 1e-9)
}

func TestGeminiOmniBilling_CatalogPolicySafetyNet(t *testing.T) {
	ps := &PricingService{}
	parsed, err := ps.parsePricingData([]byte(`{
		"gemini-omni-1.2-flash": {
			"input_cost_per_token": 1.5e-6,
			"output_cost_per_token": 9e-6,
			"litellm_provider": "gemini"
		}
	}`))
	require.NoError(t, err)
	ps.pricingData = parsed

	svc := NewBillingService(nil, ps)

	p, err := svc.getModelPricingAt("gemini-omni-1.2-flash", time.Date(2026, 11, 14, 0, 0, 0, 0, time.UTC))
	require.NoError(t, err)
	require.InDelta(t, 12e-6, p.VideoOutputPricePerToken, 1e-12)
}

func TestGeminiOmniBilling_HonorsChannelAndIntervalOutputOverride(t *testing.T) {
	svc := NewBillingService(nil, nil)

	// 1. Channel OutputPrice: 0 must make video output free ($0) instead of charging $12/MTok.
	zero := 0.0
	chZero, err := svc.GetModelPricingWithChannel("gemini-omni-1.2-flash", &ChannelModelPricing{
		OutputPrice: &zero,
	})
	require.NoError(t, err)
	bdZero := svc.computeTokenBreakdown(chZero, UsageTokens{
		InputTokens:       1000,
		OutputTokens:      1_000_000,
		VideoOutputTokens: 1_000_000,
	}, 1.0, "", true)
	require.Zero(t, bdZero.OutputCost)

	// 2. Channel custom non-zero OutputPrice scales VideoOutputPricePerToken proportionally (2x text -> 2x video).
	customOut := 18e-6
	chScaled, err := svc.GetModelPricingWithChannel("gemini-omni-1.2-flash", &ChannelModelPricing{
		OutputPrice: &customOut,
	})
	require.NoError(t, err)
	require.InDelta(t, 24e-6, chScaled.VideoOutputPricePerToken, 1e-12)

	// 3. Interval OutputPrice / OutputMultiplier updates VideoOutputPricePerToken cleanly.
	base, err := svc.GetModelPricing("gemini-omni-1.2-flash")
	require.NoError(t, err)
	ivZero := intervalToModelPricing(&PricingInterval{OutputPrice: &zero}, base, nil)
	require.Zero(t, ivZero.VideoOutputPricePerToken)

	half := 0.5
	ivMult := intervalToModelPricing(&PricingInterval{OutputMultiplier: &half}, base, nil)
	require.InDelta(t, 4.5e-6, ivMult.OutputPricePerToken, 1e-12)
	require.InDelta(t, 6e-6, ivMult.VideoOutputPricePerToken, 1e-12)
}

func TestGemini38TTSBilling_RatesCutoffPriorityAndFlex(t *testing.T) {
	svc := NewBillingService(nil, nil)

	pre2027 := time.Date(2026, 11, 14, 12, 0, 0, 0, time.UTC)
	at2027 := time.Date(2027, 1, 1, 0, 0, 0, 0, time.UTC)

	// Flash-Lite TTS: 2026 ($0.50 / $6.00) vs 2027 ($1.00 / $12.00), Priority 1.8x, Flex 0.5x
	pLitePre, err := svc.getModelPricingAt("gemini-3.8-flash-lite-tts", pre2027)
	require.NoError(t, err)
	require.InDelta(t, 0.5e-6, pLitePre.InputPricePerToken, 1e-12)
	require.InDelta(t, 6e-6, pLitePre.AudioOutputPricePerToken, 1e-12)

	stdPre := svc.computeTokenBreakdown(pLitePre, UsageTokens{
		InputTokens:       1_000_000,
		OutputTokens:      1_000_000,
		AudioOutputTokens: 1_000_000,
	}, 1.0, "", true)
	require.InDelta(t, 6.5, stdPre.TotalCost, 1e-9)

	prioPre := svc.computeTokenBreakdown(pLitePre, UsageTokens{
		InputTokens:       1_000_000,
		OutputTokens:      1_000_000,
		AudioOutputTokens: 1_000_000,
	}, 1.0, "priority", true)
	require.InDelta(t, 6.5*1.8, prioPre.TotalCost, 1e-9)

	flexPre := svc.computeTokenBreakdown(pLitePre, UsageTokens{
		InputTokens:       1_000_000,
		OutputTokens:      1_000_000,
		AudioOutputTokens: 1_000_000,
	}, 1.0, "flex", true)
	require.InDelta(t, 6.5*0.5, flexPre.TotalCost, 1e-9)

	pLite2027, err := svc.getModelPricingAt("gemini-3.8-flash-lite-tts", at2027)
	require.NoError(t, err)
	require.InDelta(t, 1e-6, pLite2027.InputPricePerToken, 1e-12)
	require.InDelta(t, 12e-6, pLite2027.AudioOutputPricePerToken, 1e-12)

	// Flash TTS: 2026 ($0.50 / $9.00) vs 2027 ($1.00 / $18.00)
	pFlashPre, err := svc.getModelPricingAt("gemini-3.8-flash-tts", pre2027)
	require.NoError(t, err)
	require.InDelta(t, 0.5e-6, pFlashPre.InputPricePerToken, 1e-12)
	require.InDelta(t, 9e-6, pFlashPre.AudioOutputPricePerToken, 1e-12)

	pFlash2027, err := svc.getModelPricingAt("gemini-3.8-flash-tts", at2027)
	require.NoError(t, err)
	require.InDelta(t, 1e-6, pFlash2027.InputPricePerToken, 1e-12)
	require.InDelta(t, 18e-6, pFlash2027.AudioOutputPricePerToken, 1e-12)

	// Base gemini-3.8-flash must still use its own prices and not TTS policy.
	pBaseFlash, err := svc.getModelPricingAt("gemini-3.8-flash", pre2027)
	require.NoError(t, err)
	require.Zero(t, pBaseFlash.AudioOutputPricePerToken)

	// Legacy / 3.1 TTS fallback checks
	p25Flash, err := svc.getModelPricingAt("gemini-2.5-flash-preview-tts", pre2027)
	require.NoError(t, err)
	require.InDelta(t, 0.5e-6, p25Flash.InputPricePerToken, 1e-12)
	require.InDelta(t, 10e-6, p25Flash.AudioOutputPricePerToken, 1e-12)

	p25Pro, err := svc.getModelPricingAt("gemini-2.5-pro-preview-tts", pre2027)
	require.NoError(t, err)
	require.InDelta(t, 1e-6, p25Pro.InputPricePerToken, 1e-12)
	require.InDelta(t, 20e-6, p25Pro.AudioOutputPricePerToken, 1e-12)

	p31Flash, err := svc.getModelPricingAt("gemini-3.1-flash-tts-preview", pre2027)
	require.NoError(t, err)
	require.InDelta(t, 1e-6, p31Flash.InputPricePerToken, 1e-12)
	require.InDelta(t, 20e-6, p31Flash.AudioOutputPricePerToken, 1e-12)
}

func TestGeminiUnpricedModelsAndBundledCatalog(t *testing.T) {
	catalogPath := filepath.Join("..", "..", "resources", "model-pricing", "model_prices_and_context_window.json")
	data, err := os.ReadFile(catalogPath)
	require.NoError(t, err)

	ps := &PricingService{}
	parsed, err := ps.parsePricingData(data)
	require.NoError(t, err)
	ps.pricingData = parsed

	for _, model := range []string{
		"aqa",
		"gemini-2.5-flash-preview-tts",
		"gemini-2.5-pro-preview-tts",
		"gemini-3.1-flash-tts-preview",
		"gemini-3.8-flash-lite-tts",
		"gemini-3.8-flash-tts",
		"gemini-nano-banana-2.1",
		"gemini-omni-1.1-flash",
		"gemini-omni-1.2-flash",
		"gemini-omni-flash-preview",
		"gemma-4-26b-a4b-it",
		"gemma-4-31b-it",
		"lyria-realtime-exp",
		"nano-banana-pro-preview",
	} {
		require.NotNilf(t, ps.GetModelPricing(model), "expected catalog pricing for %s", model)
	}

	svc := NewBillingService(nil, ps)

	// nano-banana-pro-preview resolves to gemini-3-pro-image-preview rates.
	nanoPro, err := svc.GetModelPricing("nano-banana-pro-preview")
	require.NoError(t, err)
	require.InDelta(t, 2e-6, nanoPro.InputPricePerToken, 1e-12)
	require.InDelta(t, 120e-6, nanoPro.ImageOutputPricePerToken, 1e-12)

	// Free models have explicit $0 entries and return $0 cost.
	for _, freeModel := range []string{"gemma-4-26b-a4b-it", "gemma-4-31b-it", "aqa", "lyria-realtime-exp"} {
		p, err := svc.GetModelPricing(freeModel)
		require.NoError(t, err, freeModel)
		bd := svc.computeTokenBreakdown(p, UsageTokens{InputTokens: 1000, OutputTokens: 1000}, 1.0, "", true)
		require.Zero(t, bd.TotalCost, freeModel)
	}
}
