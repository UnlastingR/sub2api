// Package gemini provides minimal fallback model metadata for Gemini native endpoints.
// It is used when upstream model listing is unavailable (e.g. OAuth token missing AI Studio scopes).
package gemini

import "strings"

type Model struct {
	Name                       string   `json:"name"`
	DisplayName                string   `json:"displayName,omitempty"`
	Description                string   `json:"description,omitempty"`
	SupportedGenerationMethods []string `json:"supportedGenerationMethods,omitempty"`
}

type ModelsListResponse struct {
	Models []Model `json:"models"`
}

func DefaultModels() []Model {
	methods := []string{"generateContent", "streamGenerateContent"}
	return []Model{
		{Name: "models/gemini-3.1-flash-image", SupportedGenerationMethods: methods},
		{Name: "models/gemini-3.1-flash-image-preview", SupportedGenerationMethods: methods},
		{Name: "models/gemini-3.1-flash-lite-image", SupportedGenerationMethods: methods},
		{Name: "models/gemini-3-pro-image", SupportedGenerationMethods: methods},
		{Name: "models/gemini-3-pro-image-preview", SupportedGenerationMethods: methods},
		{Name: "models/gemini-3-flash", SupportedGenerationMethods: methods},
		{Name: "models/gemini-3-flash-preview", SupportedGenerationMethods: methods},
		{Name: "models/gemini-3.1-pro-preview", SupportedGenerationMethods: methods},
		{Name: "models/gemini-3.1-pro-preview-customtools", SupportedGenerationMethods: methods},
		{Name: "models/gemini-3.1-flash-lite", SupportedGenerationMethods: methods},
		{Name: "models/gemini-3.1-flash-lite-preview", SupportedGenerationMethods: methods},
		{Name: "models/gemini-3.5-flash", SupportedGenerationMethods: methods},
		{Name: "models/gemini-3.5-flash-lite", SupportedGenerationMethods: methods},
		{Name: "models/gemini-3.6-flash", SupportedGenerationMethods: methods},
		{Name: "models/gemini-3.7-flash", SupportedGenerationMethods: methods},
		{Name: "models/gemini-3.8-flash", SupportedGenerationMethods: methods},
		{Name: "models/gemini-4-argon", SupportedGenerationMethods: methods},
		{Name: "models/gemini-4-carbon", SupportedGenerationMethods: methods},
		{Name: "models/gemini-4-hydrogen", SupportedGenerationMethods: methods},
		{Name: "models/gemini-decision-1", SupportedGenerationMethods: methods},
		{Name: "models/gemini-2.5-computer-use-preview", SupportedGenerationMethods: methods},
		{Name: "models/gemini-2.5-computer-use-preview-10-2025", SupportedGenerationMethods: methods},
		{Name: "models/gemini-robotics-er-1.5-preview", SupportedGenerationMethods: methods},
		{Name: "models/gemini-robotics-er-1.6-preview", SupportedGenerationMethods: methods},
		{Name: "models/gemini-robotics-er-2-preview", SupportedGenerationMethods: methods},
		{Name: "models/gemini-robotics-er-2-streaming-preview", SupportedGenerationMethods: []string{"bidiGenerateContent"}},
		{Name: "models/gemini-3.8-live", SupportedGenerationMethods: []string{"bidiGenerateContent"}},
		{Name: "models/gemini-3.8-live-extended-thinking", SupportedGenerationMethods: []string{"bidiGenerateContent"}},
		{Name: "models/gemini-3.5-live-translate-preview", SupportedGenerationMethods: []string{"bidiGenerateContent"}},
		{Name: "models/gemini-3.5-transcribe-live", SupportedGenerationMethods: []string{"bidiGenerateContent"}},
		{Name: "models/antigravity-preview-09-2026", SupportedGenerationMethods: []string{"interactions"}},
		{Name: "models/deep-research-preview-04-2026", SupportedGenerationMethods: []string{"interactions"}},
		{Name: "models/deep-research-max-preview-04-2026", SupportedGenerationMethods: []string{"interactions"}},
		{Name: "models/gemini-embedding-001", SupportedGenerationMethods: []string{"embedContent", "batchEmbedContents", "asyncBatchEmbedContent"}},
		{Name: "models/gemini-embedding-2", SupportedGenerationMethods: []string{"embedContent", "batchEmbedContents", "asyncBatchEmbedContent"}},
	}
}

func HasFallbackModel(model string) bool {
	trimmed := strings.TrimSpace(model)
	if trimmed == "" {
		return false
	}
	if !strings.HasPrefix(trimmed, "models/") {
		trimmed = "models/" + trimmed
	}
	for _, model := range DefaultModels() {
		if model.Name == trimmed {
			return true
		}
	}
	return false
}

func FallbackModelsList() ModelsListResponse {
	return ModelsListResponse{Models: DefaultModels()}
}

func FallbackModel(model string) Model {
	methods := []string{"generateContent", "streamGenerateContent"}
	if model == "" {
		return Model{Name: "models/unknown", SupportedGenerationMethods: methods}
	}
	name := model
	if !strings.HasPrefix(name, "models/") {
		name = "models/" + name
	}
	for _, candidate := range DefaultModels() {
		if candidate.Name == name {
			return candidate
		}
	}
	return Model{Name: name, SupportedGenerationMethods: methods}
}
