package gemini

import "testing"

func TestDefaultModels_ContainsFallbackCatalogModels(t *testing.T) {
	t.Parallel()

	models := DefaultModels()
	byName := make(map[string]Model, len(models))
	for _, model := range models {
		byName[model.Name] = model
	}

	required := []string{
		"models/gemini-3-pro-image",
		"models/gemini-3.1-pro-preview-customtools",
		"models/gemini-3.1-flash-image",
	}

	for _, name := range required {
		model, ok := byName[name]
		if !ok {
			t.Fatalf("expected fallback model %q to exist", name)
		}
		if len(model.SupportedGenerationMethods) == 0 {
			t.Fatalf("expected fallback model %q to advertise generation methods", name)
		}
	}
}

func TestHasFallbackModel_RecognizesCustomtoolsModel(t *testing.T) {
	t.Parallel()

	if !HasFallbackModel("gemini-3.1-pro-preview-customtools") {
		t.Fatalf("expected customtools model to exist in fallback catalog")
	}
	if !HasFallbackModel("models/gemini-3.1-pro-preview-customtools") {
		t.Fatalf("expected prefixed customtools model to exist in fallback catalog")
	}
	if HasFallbackModel("gemini-unknown") {
		t.Fatalf("did not expect unknown model to exist in fallback catalog")
	}
}

func TestFallbackModel_PreservesDeclaredGenerationMethods(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name, method string
	}{
		{"gemini-3.8-live", "bidiGenerateContent"},
		{"models/deep-research-preview-04-2026", "interactions"},
		{"gemini-embedding-2", "embedContent"},
	} {
		got := FallbackModel(tc.name)
		found := false
		for _, method := range got.SupportedGenerationMethods {
			if method == tc.method {
				found = true
			}
		}
		if !found {
			t.Errorf("%s: expected method %s, got %v", tc.name, tc.method, got.SupportedGenerationMethods)
		}
	}
}
