//go:build unit

package service

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type interactionTestCacheStub struct {
	mu       sync.Mutex
	bindings map[string]int64
	pending  map[string][]byte
	claims   map[string]bool
}

func (c *interactionTestCacheStub) GetSessionAccountID(_ context.Context, _ int64, sessionHash string) (int64, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if id, ok := c.bindings[sessionHash]; ok {
		return id, nil
	}
	return 0, ErrStickySessionNotFound
}

func (c *interactionTestCacheStub) SetSessionAccountID(_ context.Context, _ int64, sessionHash string, accountID int64, _ time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.bindings == nil {
		c.bindings = make(map[string]int64)
	}
	c.bindings[sessionHash] = accountID
	return nil
}

func (c *interactionTestCacheStub) RefreshSessionTTL(context.Context, int64, string, time.Duration) error {
	return nil
}

func (c *interactionTestCacheStub) DeleteSessionAccountID(_ context.Context, _ int64, sessionHash string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.bindings, sessionHash)
	return nil
}

func (c *interactionTestCacheStub) SetGrokVideoPendingBilling(_ context.Context, key string, payload []byte, _ time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.pending == nil {
		c.pending = make(map[string][]byte)
	}
	c.pending[key] = append([]byte(nil), payload...)
	return nil
}

func (c *interactionTestCacheStub) GetGrokVideoPendingBilling(_ context.Context, key string) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if v, ok := c.pending[key]; ok {
		return append([]byte(nil), v...), nil
	}
	return nil, nil
}

func (c *interactionTestCacheStub) ClaimGrokVideoBilled(_ context.Context, key string, _ time.Duration) (bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.claims == nil {
		c.claims = make(map[string]bool)
	}
	if c.claims[key] {
		return false, nil
	}
	c.claims[key] = true
	return true, nil
}

func (c *interactionTestCacheStub) ReleaseGrokVideoBilled(_ context.Context, key string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.claims, key)
	return nil
}

func (c *interactionTestCacheStub) SetReasoningContent(context.Context, string, string, time.Duration) error {
	return nil
}

func (c *interactionTestCacheStub) GetReasoningContent(context.Context, string) (string, error) {
	return "", nil
}

func TestIsSafeGeminiInteractionID(t *testing.T) {
	t.Parallel()

	require.True(t, IsSafeGeminiInteractionID("v1_ChdPU0F4YWFtNkFwS2kxZThQZ05lbXdROBIXT1NBeGFhbTZBcEtpMWU4UGdOZW13UTg"))
	require.True(t, IsSafeGeminiInteractionID("int_123-abc.xyz"))
	require.False(t, IsSafeGeminiInteractionID(""))
	require.False(t, IsSafeGeminiInteractionID("."))
	require.False(t, IsSafeGeminiInteractionID(".."))
	require.False(t, IsSafeGeminiInteractionID("abc/def"))
	require.False(t, IsSafeGeminiInteractionID("abc?stream=true"))
	require.False(t, IsSafeGeminiInteractionID(strings.Repeat("a", 257)))
}

func TestIsGeminiAIStudioEndpointAccount(t *testing.T) {
	t.Parallel()

	require.True(t, IsGeminiAIStudioEndpointAccount(&Account{
		Platform:    PlatformGemini,
		Type:        AccountTypeAPIKey,
		Credentials: map[string]any{"api_key": "sk-test"},
	}))
	require.False(t, IsGeminiAIStudioEndpointAccount(&Account{
		Platform:    PlatformGemini,
		Type:        AccountTypeAPIKey,
		Credentials: map[string]any{},
	}))
	require.True(t, IsGeminiAIStudioEndpointAccount(&Account{
		Platform: PlatformGemini,
		Type:     AccountTypeOAuth,
		Credentials: map[string]any{
			"oauth_type": "ai_studio",
		},
	}))
	require.False(t, IsGeminiAIStudioEndpointAccount(&Account{
		Platform: PlatformGemini,
		Type:     AccountTypeOAuth,
		Credentials: map[string]any{
			"oauth_type": "code_assist",
			"project_id": "proj-1",
		},
	}))
	require.False(t, IsGeminiAIStudioEndpointAccount(&Account{
		Platform: PlatformGemini,
		Type:     AccountTypeOAuth,
		Credentials: map[string]any{
			"project_id": "proj-2",
		},
	}))
	require.False(t, IsGeminiAIStudioEndpointAccount(&Account{
		Platform: PlatformGemini,
		Type:     AccountTypeOAuth,
		Credentials: map[string]any{
			"oauth_type": "google_one",
		},
	}))
	require.False(t, IsGeminiAIStudioEndpointAccount(&Account{
		Platform: PlatformGemini,
		Type:     AccountTypeServiceAccount,
	}))
	require.False(t, IsGeminiAIStudioEndpointAccount(&Account{
		Platform: PlatformAntigravity,
		Type:     AccountTypeOAuth,
	}))
}

func TestForwardDecisions_NonStreamingAndUsage(t *testing.T) {
	gin.SetMode(gin.TestMode)
	upstream := &httpUpstreamRecorder{
		resp: &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"req-dec-1"}},
			Body: io.NopCloser(strings.NewReader(
				`{"answers":{"need_search":{"answer":true}},"usageMetadata":{"promptTokenCount":42,"candidatesTokenCount":0,"totalTokenCount":42}}`,
			)),
		},
	}
	svc := &GeminiMessagesCompatService{
		httpUpstream: upstream,
		cfg:          &config.Config{},
	}
	account := &Account{
		ID:       10,
		Platform: PlatformGemini,
		Type:     AccountTypeAPIKey,
		Credentials: map[string]any{
			"api_key":  "sk-goog",
			"base_url": "https://relay.example.com",
		},
	}

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1beta/decisions", nil)

	body := []byte(`{"model":"gemini-decision-1","input":"hello","questions":{"need_search":{"question":"search?"}}}`)
	res, err := svc.ForwardDecisions(context.Background(), c, account, "gemini-decision-1", "gemini-decision-1-mapped", body)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, "https://relay.example.com/v1beta/decisions", upstream.lastReq.URL.String())
	require.Equal(t, "sk-goog", upstream.lastReq.Header.Get("x-goog-api-key"))
	require.Contains(t, string(upstream.lastBody), `"model":"gemini-decision-1-mapped"`)
	require.Equal(t, 42, res.Usage.InputTokens)
	require.Equal(t, 0, res.Usage.OutputTokens)
}

func TestForwardInteractionCreate_StreamingAndNonStreaming(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("non-streaming captures id and usage and rewrites mapped model", func(t *testing.T) {
		upstream := &httpUpstreamRecorder{
			resp: &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body: io.NopCloser(strings.NewReader(
					`{"id":"v1_interact_abc","model":"gemini-3.5-flash-lite","status":"completed","usage":{"total_input_tokens":100,"total_cached_tokens":20,"total_output_tokens":35,"total_thought_tokens":5}}`,
				)),
			},
		}
		svc := &GeminiMessagesCompatService{httpUpstream: upstream, cfg: &config.Config{}}
		account := &Account{
			ID:          11,
			Platform:    PlatformGemini,
			Type:        AccountTypeAPIKey,
			Credentials: map[string]any{"api_key": "sk-goog", "base_url": "https://relay.example.com"},
		}
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1beta/interactions", nil)

		// Even when originalModel equals mappedModel after composite/channel translation,
		// the raw body still holds the public alias and must be rewritten to mappedModel.
		var observed []string
		res, interactionID, err := svc.ForwardInteractionCreate(context.Background(), c, account, "gemini-3.5-flash-lite", "gemini-3.5-flash-lite", false, []byte(`{"model":"public-alias","input":"hi"}`), func(id string) {
			observed = append(observed, id)
		})
		require.NoError(t, err)
		require.Equal(t, "v1_interact_abc", interactionID)
		require.Equal(t, []string{"v1_interact_abc"}, observed)
		require.Equal(t, 80, res.Usage.InputTokens)
		require.Equal(t, 20, res.Usage.CacheReadInputTokens)
		require.Equal(t, 40, res.Usage.OutputTokens)
		require.Equal(t, "https://relay.example.com/v1beta/interactions", upstream.lastReq.URL.String())
		require.Contains(t, string(upstream.lastBody), `"model":"gemini-3.5-flash-lite"`)
	})

	t.Run("agent body is rewritten to mapped model", func(t *testing.T) {
		rewritten := rewriteGeminiAIStudioBodyModel([]byte(`{"agent":"public-agent-alias","input":"research"}`), "deep-research-pro-preview-12-2025")
		require.Contains(t, string(rewritten), `"agent":"deep-research-pro-preview-12-2025"`)
	})

	t.Run("image generation gate blocks interactions when disallowed", func(t *testing.T) {
		svc := &GeminiMessagesCompatService{cfg: &config.Config{}}
		account := &Account{ID: 11, Platform: PlatformGemini, Type: AccountTypeAPIKey, Credentials: map[string]any{"api_key": "sk-goog"}}
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1beta/interactions", nil)
		c.Set("api_key", &APIKey{
			ID:    1,
			Group: &Group{ID: 1, Platform: PlatformGemini, AllowImageGeneration: false},
		})

		_, _, err := svc.ForwardInteractionCreate(context.Background(), c, account, "gemini-4-hydrogen", "gemini-4-hydrogen", false, []byte(`{"model":"gemini-4-hydrogen","input":"draw a cat","response_modalities":["IMAGE"]}`), nil)
		require.Error(t, err)
		require.Equal(t, http.StatusBadRequest, rec.Code)
	})

	t.Run("streaming SSE captures interaction id eagerly on start and final usage", func(t *testing.T) {
		ssePayload := strings.Join([]string{
			`event: interaction.start`,
			`data: {"event_type":"interaction.start","interaction":{"id":"v1_stream_1","model":"gemini-4-hydrogen"}}`,
			``,
			`event: interaction.completed`,
			`data: {"event_type":"interaction.completed","interaction":{"id":"v1_stream_1","model":"gemini-4-hydrogen","usage":{"total_input_tokens":50,"total_output_tokens":12}}}`,
			``,
		}, "\n")
		upstream := &httpUpstreamRecorder{
			resp: &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       io.NopCloser(strings.NewReader(ssePayload)),
			},
		}
		svc := &GeminiMessagesCompatService{httpUpstream: upstream, cfg: &config.Config{}}
		account := &Account{
			ID:          12,
			Platform:    PlatformGemini,
			Type:        AccountTypeAPIKey,
			Credentials: map[string]any{"api_key": "sk-goog", "base_url": "https://relay.example.com"},
		}
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1beta/interactions", nil)

		var earlyIDs []string
		res, interactionID, err := svc.ForwardInteractionCreate(context.Background(), c, account, "gemini-4-hydrogen", "gemini-4-hydrogen", true, []byte(`{"model":"gemini-4-hydrogen","input":"hi","stream":true}`), func(id string) {
			earlyIDs = append(earlyIDs, id)
		})
		require.NoError(t, err)
		require.Equal(t, "v1_stream_1", interactionID)
		require.Equal(t, []string{"v1_stream_1"}, earlyIDs)
		require.Equal(t, 50, res.Usage.InputTokens)
		require.Equal(t, 12, res.Usage.OutputTokens)
		require.True(t, res.Stream)
		require.Equal(t, "text/event-stream", upstream.lastReq.Header.Get("Accept"))
		require.Contains(t, rec.Body.String(), `"event_type":"interaction.completed"`)
	})
}

func TestForwardInteractionByID_AndBindingIsolation(t *testing.T) {
	gin.SetMode(gin.TestMode)
	cache := &interactionTestCacheStub{}
	upstream := &httpUpstreamRecorder{
		responses: []*http.Response{
			{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"id":"v1_bg_1","model":"deep-research-preview-04-2026","status":"completed","usage":{"total_input_tokens":200,"total_output_tokens":60}}`)),
			},
			{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"id":"v1_bg_1","status":"cancelled"}`)),
			},
			{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{}`)),
			},
		},
	}
	svc := &GeminiMessagesCompatService{
		httpUpstream: upstream,
		cache:        cache,
		cfg:          &config.Config{},
	}
	gid := int64(5)
	require.NoError(t, svc.BindGeminiInteractionAccount(context.Background(), &gid, 100, "v1_bg_1", 77))

	acctID, verified, err := svc.ResolveGeminiInteractionAccount(context.Background(), &gid, 100, "v1_bg_1")
	require.NoError(t, err)
	require.True(t, verified)
	require.Equal(t, int64(77), acctID)

	acctID2, verified2, err := svc.ResolveGeminiInteractionAccount(context.Background(), &gid, 200, "v1_bg_1")
	require.NoError(t, err)
	require.False(t, verified2)
	require.Equal(t, int64(0), acctID2)

	acctID3, verified3, err := svc.ResolveGeminiInteractionAccount(context.Background(), &gid, 100, "v1_missing")
	require.NoError(t, err)
	require.False(t, verified3)
	require.Equal(t, int64(0), acctID3)

	metaIn := GeminiInteractionMeta{
		RequestModel:      "public-alias",
		UpstreamModel:     "deep-research-preview-04-2026",
		PricingAtUnixNano: 123456789,
		ForceCacheBilling: true,
		ChannelUsageFields: ChannelUsageFields{
			ChannelID:          88,
			OriginalModel:      "public-alias",
			ChannelMappedModel: "deep-research-preview-04-2026",
			BillingModelSource: "requested",
		},
	}
	require.NoError(t, svc.SaveGeminiInteractionMeta(context.Background(), "v1_bg_1", metaIn))
	metaOut, ok, err := svc.GetGeminiInteractionMeta(context.Background(), "v1_bg_1")
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, metaIn, *metaOut)

	claimed, err := svc.ClaimGeminiInteractionBilled(context.Background(), "v1_bg_1")
	require.NoError(t, err)
	require.True(t, claimed)
	claimedAgain, err := svc.ClaimGeminiInteractionBilled(context.Background(), "v1_bg_1")
	require.NoError(t, err)
	require.False(t, claimedAgain)

	account := &Account{
		ID:          77,
		Platform:    PlatformGemini,
		Type:        AccountTypeAPIKey,
		Credentials: map[string]any{"api_key": "sk-goog", "base_url": "https://relay.example.com"},
	}

	rec1 := httptest.NewRecorder()
	c1, _ := gin.CreateTestContext(rec1)
	c1.Request = httptest.NewRequest(http.MethodGet, "/v1beta/interactions/v1_bg_1?last_event_id=evt_1", nil)
	res1, observedModel, err := svc.ForwardInteractionByID(context.Background(), c1, account, http.MethodGet, "v1_bg_1", false, url.Values{"last_event_id": {"evt_1"}, "key": {"evil"}}, false)
	require.NoError(t, err)
	require.Equal(t, "deep-research-preview-04-2026", observedModel)
	require.Equal(t, "deep-research-preview-04-2026", res1.UpstreamResponseModel)
	require.False(t, res1.UpstreamResponseModelConflict)
	require.Equal(t, 200, res1.Usage.InputTokens)
	require.Equal(t, "https://relay.example.com/v1beta/interactions/v1_bg_1?last_event_id=evt_1", upstream.requests[0].URL.String())

	rec2 := httptest.NewRecorder()
	c2, _ := gin.CreateTestContext(rec2)
	c2.Request = httptest.NewRequest(http.MethodPost, "/v1beta/interactions/v1_bg_1/cancel", nil)
	_, _, err = svc.ForwardInteractionByID(context.Background(), c2, account, http.MethodPost, "v1_bg_1", true, nil, false)
	require.NoError(t, err)
	require.Equal(t, http.MethodPost, upstream.requests[1].Method)
	require.Equal(t, "https://relay.example.com/v1beta/interactions/v1_bg_1/cancel", upstream.requests[1].URL.String())

	rec3 := httptest.NewRecorder()
	c3, _ := gin.CreateTestContext(rec3)
	c3.Request = httptest.NewRequest(http.MethodDelete, "/v1beta/interactions/v1_bg_1", nil)
	_, _, err = svc.ForwardInteractionByID(context.Background(), c3, account, http.MethodDelete, "v1_bg_1", false, nil, false)
	require.NoError(t, err)
	require.Equal(t, http.MethodDelete, upstream.requests[2].Method)
	require.Equal(t, "https://relay.example.com/v1beta/interactions/v1_bg_1", upstream.requests[2].URL.String())
}

func TestGeminiInteractions_ImageCountsModelObservationAndSnapshotPolling(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("non-streaming counts outputs images and observes interaction model", func(t *testing.T) {
		upstream := &httpUpstreamRecorder{
			resp: &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body: io.NopCloser(strings.NewReader(
					`{"id":"v1_img_1","model":"gemini-3-pro-image-preview","status":"completed","outputs":[{"type":"image","data":"AAA","mime_type":"image/png"}]}`,
				)),
			},
		}
		svc := &GeminiMessagesCompatService{httpUpstream: upstream, cfg: &config.Config{}}
		account := &Account{ID: 10, Platform: PlatformGemini, Type: AccountTypeAPIKey, Credentials: map[string]any{"api_key": "sk-goog"}}
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1beta/interactions", nil)

		res, _, err := svc.ForwardInteractionCreate(context.Background(), c, account, "gemini-3-pro-image-preview", "gemini-3-pro-image-preview", false, []byte(`{"model":"gemini-3-pro-image-preview","input":"draw"}`), nil)
		require.NoError(t, err)
		require.True(t, res.ImageEndpoint)
		require.Equal(t, 1, res.ImageCount)
		require.Equal(t, "gemini-3-pro-image-preview", res.UpstreamResponseModel)
		require.False(t, res.UpstreamResponseModelConflict)
	})

	t.Run("SSE image deltas and completed snapshot do not double count", func(t *testing.T) {
		ssePayload := strings.Join([]string{
			`event: interaction.start`,
			`data: {"event_type":"interaction.start","interaction":{"id":"v1_img_sse","model":"gemini-3-pro-image-preview"}}`,
			``,
			`event: content.delta`,
			`data: {"event_type":"content.delta","delta":{"type":"image","data":"AAA","mime_type":"image/png"}}`,
			``,
			`event: content.delta`,
			`data: {"event_type":"content.delta","delta":{"type":"image","uri":"https://example.com/2.png"}}`,
			``,
			`event: interaction.completed`,
			`data: {"event_type":"interaction.completed","interaction":{"id":"v1_img_sse","model":"gemini-3-pro-image-preview","outputs":[{"type":"image","data":"AAA"},{"type":"image","uri":"https://example.com/2.png"}]}}`,
			``,
		}, "\n")
		upstream := &httpUpstreamRecorder{
			resp: &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       io.NopCloser(strings.NewReader(ssePayload)),
			},
		}
		svc := &GeminiMessagesCompatService{httpUpstream: upstream, cfg: &config.Config{}}
		account := &Account{ID: 11, Platform: PlatformGemini, Type: AccountTypeAPIKey, Credentials: map[string]any{"api_key": "sk-goog"}}
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1beta/interactions", nil)

		res, _, err := svc.ForwardInteractionCreate(context.Background(), c, account, "gemini-3-pro-image-preview", "gemini-3-pro-image-preview", true, []byte(`{"model":"gemini-3-pro-image-preview","input":"draw two","stream":true}`), nil)
		require.NoError(t, err)
		require.True(t, res.ImageEndpoint)
		require.Equal(t, 2, res.ImageCount)
		require.Equal(t, "gemini-3-pro-image-preview", res.UpstreamResponseModel)
		require.False(t, res.UpstreamResponseModelConflict)
	})

	t.Run("FetchGeminiInteractionSnapshot parses progress and terminal usage", func(t *testing.T) {
		upstream := &httpUpstreamRecorder{
			responses: []*http.Response{
				{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"application/json"}},
					Body:       io.NopCloser(strings.NewReader(`{"id":"v1_snap_1","status":"in_progress","agent":"deep-research-pro"}`)),
				},
				{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"application/json"}},
					Body:       io.NopCloser(strings.NewReader(`{"id":"v1_snap_1","status":"completed","agent":"deep-research-pro","usage":{"total_input_tokens":300,"total_output_tokens":90}}`)),
				},
			},
		}
		svc := &GeminiMessagesCompatService{httpUpstream: upstream, cfg: &config.Config{}}
		account := &Account{ID: 12, Platform: PlatformGemini, Type: AccountTypeAPIKey, Credentials: map[string]any{"api_key": "sk-goog"}}

		res1, status1, terminal1, err := svc.FetchGeminiInteractionSnapshot(context.Background(), account, "v1_snap_1")
		require.NoError(t, err)
		require.Equal(t, "in_progress", status1)
		require.False(t, terminal1)
		require.False(t, res1.InteractionTerminal)
		require.Zero(t, res1.Usage.InputTokens)

		res2, status2, terminal2, err := svc.FetchGeminiInteractionSnapshot(context.Background(), account, "v1_snap_1")
		require.NoError(t, err)
		require.Equal(t, "completed", status2)
		require.True(t, terminal2)
		require.True(t, res2.InteractionTerminal)
		require.Equal(t, 300, res2.Usage.InputTokens)
		require.Equal(t, 90, res2.Usage.OutputTokens)
		require.Equal(t, "deep-research-pro", res2.UpstreamResponseModel)
	})

	t.Run("in_progress create with partial usage does not mark InteractionTerminal", func(t *testing.T) {
		upstream := &httpUpstreamRecorder{
			resp: &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body: io.NopCloser(strings.NewReader(
					`{"id":"v1_prog_1","model":"gemini-4-hydrogen","status":"in_progress","usage":{"total_input_tokens":50}}`,
				)),
			},
		}
		svc := &GeminiMessagesCompatService{httpUpstream: upstream, cfg: &config.Config{}}
		account := &Account{ID: 13, Platform: PlatformGemini, Type: AccountTypeAPIKey, Credentials: map[string]any{"api_key": "sk-goog"}}
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1beta/interactions", nil)

		res, id, err := svc.ForwardInteractionCreate(context.Background(), c, account, "gemini-4-hydrogen", "gemini-4-hydrogen", false, []byte(`{"model":"gemini-4-hydrogen","input":"deep","background":true}`), nil)
		require.NoError(t, err)
		require.Equal(t, "v1_prog_1", id)
		require.False(t, res.InteractionTerminal)
	})

	t.Run("ApplyGeminiInteractionMetaModels unconditionally restores creation models while preserving UpstreamResponseModel", func(t *testing.T) {
		res := &ForwardResult{
			Model:                 "gemini-3.5-flash-lite-001",
			UpstreamModel:         "gemini-3.5-flash-lite-001",
			UpstreamResponseModel: "gemini-3.5-flash-lite-001",
			Usage: ClaudeUsage{
				InputTokens:  100,
				OutputTokens: 25,
			},
		}
		meta := &GeminiInteractionMeta{
			RequestModel:  "client-alias",
			UpstreamModel: "gemini-3-pro-image-preview",
		}
		ApplyGeminiInteractionMetaModels(res, "gemini-3.5-flash-lite-001", meta)
		require.Equal(t, "client-alias", res.Model)
		require.Equal(t, "gemini-3-pro-image-preview", res.UpstreamModel)
		require.Equal(t, "gemini-3.5-flash-lite-001", res.UpstreamResponseModel)
		require.Equal(t, 1, res.ImageCount)
		require.True(t, res.ImageEndpoint)
	})

	t.Run("in-memory pending queue schedules lists due and removes on settle", func(t *testing.T) {
		cache := &interactionTestCacheStub{}
		svc := &GeminiMessagesCompatService{cache: cache}
		now := time.Now()
		require.NoError(t, svc.ScheduleGeminiInteractionPending(context.Background(), "v1_due_1", now.Add(-time.Second)))
		require.NoError(t, svc.ScheduleGeminiInteractionPending(context.Background(), "v1_future_1", now.Add(time.Minute)))

		due, err := svc.ListDueGeminiInteractionPending(context.Background(), now, 10)
		require.NoError(t, err)
		require.Equal(t, []string{"v1_due_1"}, due)

		require.NoError(t, svc.MarkGeminiInteractionSettled(context.Background(), "v1_due_1"))
		settled, err := svc.IsGeminiInteractionSettled(context.Background(), "v1_due_1")
		require.NoError(t, err)
		require.True(t, settled)

		dueAfter, err := svc.ListDueGeminiInteractionPending(context.Background(), now, 10)
		require.NoError(t, err)
		require.Empty(t, dueAfter)
	})

	t.Run("ClaimGeminiInteractionBilled uses short recoverable lease and blocks after settlement", func(t *testing.T) {
		var claimedTTL time.Duration
		cache := &interactionTestCacheStub{}
		svc := &GeminiMessagesCompatService{cache: cache}
		_ = claimedTTL

		ok, err := svc.ClaimGeminiInteractionBilled(context.Background(), "v1_lease_1")
		require.NoError(t, err)
		require.True(t, ok)

		// Simultaneous claim while lease active is blocked.
		ok2, err := svc.ClaimGeminiInteractionBilled(context.Background(), "v1_lease_1")
		require.NoError(t, err)
		require.False(t, ok2)

		// Simulating lease expiration after crash without MarkGeminiInteractionSettled allows recovery.
		cache.mu.Lock()
		delete(cache.claims, geminiInteractionBilledClaimPrefix+"v1_lease_1")
		cache.mu.Unlock()
		ok3, err := svc.ClaimGeminiInteractionBilled(context.Background(), "v1_lease_1")
		require.NoError(t, err)
		require.True(t, ok3)

		// Once settled, even after lease expiry, re-claiming returns false.
		require.NoError(t, svc.MarkGeminiInteractionSettled(context.Background(), "v1_lease_1"))
		cache.mu.Lock()
		delete(cache.claims, geminiInteractionBilledClaimPrefix+"v1_lease_1")
		cache.mu.Unlock()
		ok4, err := svc.ClaimGeminiInteractionBilled(context.Background(), "v1_lease_1")
		require.NoError(t, err)
		require.False(t, ok4)
	})
}
