//go:build unit

package service

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type extFuncHTTPUpstream struct {
	fn func(req *http.Request) (*http.Response, error)
}

func (u *extFuncHTTPUpstream) Do(req *http.Request, _ string, _ int64, _ int) (*http.Response, error) {
	return u.fn(req)
}

func (u *extFuncHTTPUpstream) DoWithTLS(req *http.Request, _ string, _ int64, _ int, _ *tlsfingerprint.Profile) (*http.Response, error) {
	return u.fn(req)
}

func TestIsAllowedGeminiAIStudioExtensionPath(t *testing.T) {
	t.Parallel()

	allowed := []string{
		"/v1beta/agents",
		"/v1beta/agents/agent-1",
		"/v1beta/environments",
		"/v1beta/environments/env-1",
		"/v1beta/environments/env-1:get",
		"/v1beta/environments/env-1:delete",
		"/v1beta/environments/env-1/files",
		"/v1beta/environments/env-1/files/.gemini/skills/my-skill/SKILL.md",
		"/upload/v1beta/environments/env-1/files/src/main.py",
		"/v1beta/credentials",
		"/v1beta/credentials/cred-1",
		"/v1beta/triggers",
		"/v1beta/triggers/trig-1",
		"/v1beta/triggers/trig-1/executions",
		"/v1beta/triggers/trig-1/executions/exec-1",
		"/v1beta/webhooks",
		"/v1beta/webhooks/wh-1",
		"/v1beta/webhooks/wh-1:ping",
		"/v1beta/webhooks/wh-1:rotateSigningSecret",
		"/v1beta/auth_tokens",
		"/v1beta/files",
		"/v1beta/files:register",
		"/v1beta/files/file-1",
		"/v1beta/files/file-1:download",
		"/upload/v1beta/files",
		"/upload/v1beta/files/file-1",
		"/download/v1beta/files/file-1",
		"/download/v1beta/files/file-1:download",
		"/v1beta/fileSearchStores",
		"/v1beta/fileSearchStores/store-1",
		"/v1beta/fileSearchStores/store-1:importFile",
		"/v1beta/fileSearchStores/store-1:uploadToFileSearchStore",
		"/v1beta/fileSearchStores/store-1/documents",
		"/v1beta/fileSearchStores/store-1/documents/doc-1",
		"/v1beta/fileSearchStores/store-1/operations/op-1",
		"/v1beta/fileSearchStores/store-1/upload/operations/op-2",
		"/upload/v1beta/fileSearchStores/store-1:uploadToFileSearchStore",
		"/v1beta/batches",
		"/v1beta/batches/b-1",
		"/v1beta/batches/b-1:cancel",
		"/v1beta/batches/b-1:updateGenerateContentBatch",
		"/v1beta/batches/b-1:updateEmbedContentBatch",
		"/v1beta/models/gemini-3.5-flash:batchGenerateContent",
		"/v1beta/models/gemini-embedding-2:asyncBatchEmbedContent",
		"/v1beta/models/gemini-embedding-2:asyncBatchEmbedContents",
		"/v1beta/models/gemini-3.5-flash/operations/op-1",
	}
	for _, p := range allowed {
		require.True(t, isAllowedGeminiAIStudioExtensionPath(p), "expected allowed path: %s", p)
	}

	blocked := []string{
		"",
		"/v1beta/agents/../secrets",
		"/v1beta/environments/env-1/files/../../etc/passwd",
		"/v1beta/environments/env-1/unknown/x",
		"/v1beta/webhooks/wh-1:evil",
		"/v1beta/files/f-1/extra",
		"/v1beta/fileSearchStores/s-1:unknown",
		"/v1beta/models/bad..model:batchGenerateContent",
		"/v1beta/models/gemini-3.5-flash:generateContent",
	}
	for _, p := range blocked {
		require.False(t, isAllowedGeminiAIStudioExtensionPath(p), "expected blocked path: %s", p)
	}
}

func TestSanitizeGeminiExtQuery(t *testing.T) {
	t.Parallel()

	in := url.Values{
		"key":            {"secret-key"},
		"api_key":        {"secret-2"},
		"x-goog-api-key": {"secret-3"},
		"access_token":   {"tok"},
		"upload_id":      {"gw-up-1"},
		"alt":            {"media"},
		"pageSize":       {"50"},
	}
	out := SanitizeGeminiExtQuery(in)
	require.Equal(t, url.Values{
		"alt":      {"media"},
		"pageSize": {"50"},
	}, out)
}

func TestExtractGeminiBatchUsage_InlineAndResponsesFile(t *testing.T) {
	t.Parallel()

	inlineJSON := []byte(`{
		"name": "batches/b-inline-1",
		"state": "BATCH_STATE_SUCCEEDED",
		"output": {
			"inlinedResponses": {
				"inlinedResponses": [
					{"response": {"usageMetadata": {"promptTokenCount": 120, "cachedContentTokenCount": 20, "candidatesTokenCount": 30, "thoughtsTokenCount": 10}}},
					{"response": {"usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 15, "candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": 258}]}}}
				]
			}
		}
	}`)

	svc := &GeminiMessagesCompatService{
		cfg:   &config.Config{Security: config.SecurityConfig{URLAllowlist: config.URLAllowlistConfig{Enabled: false, AllowInsecureHTTP: true}}},
		cache: &interactionTestCacheStub{},
	}
	usage, state, terminal, err := svc.ExtractGeminiBatchUsage(context.Background(), nil, 10, inlineJSON)
	require.NoError(t, err)
	require.Equal(t, "BATCH_STATE_SUCCEEDED", state)
	require.True(t, terminal)
	require.NotNil(t, usage)
	require.Equal(t, 150, usage.InputTokens) // (120-20) + 50
	require.Equal(t, 55, usage.OutputTokens) // (30+10) + 15
	require.Equal(t, 20, usage.CacheReadInputTokens)
	require.Equal(t, 258, usage.ImageOutputTokens)

	// External JSONL responsesFile streaming scan
	jsonlBody := strings.Join([]string{
		`{"key":"req-1","response":{"usageMetadata":{"promptTokenCount":1000,"candidatesTokenCount":200}}}`,
		`{"key":"req-2","response":{"usageMetadata":{"promptTokenCount":500,"cachedContentTokenCount":100,"candidatesTokenCount":50,"thoughtsTokenCount":25}}}`,
	}, "\n") + "\n"

	var downloadedURL string
	svc.httpUpstream = &extFuncHTTPUpstream{
		fn: func(req *http.Request) (*http.Response, error) {
			downloadedURL = req.URL.String()
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/jsonl"}},
				Body:       io.NopCloser(strings.NewReader(jsonlBody)),
			}, nil
		},
	}
	acc := &Account{
		ID:       9,
		Platform: PlatformGemini,
		Type:     AccountTypeAPIKey,
		Credentials: map[string]any{
			"api_key":  "AIza-test",
			"base_url": "https://generativelanguage.googleapis.com/v1beta",
		},
	}
	fileBatchJSON := []byte(`{
		"name": "batches/b-file-2",
		"state": "BATCH_STATE_SUCCEEDED",
		"output": {"responsesFile": "files/batch-out-2"}
	}`)
	usage2, _, terminal2, err2 := svc.ExtractGeminiBatchUsage(context.Background(), acc, 42, fileBatchJSON)
	require.NoError(t, err2)
	require.True(t, terminal2)
	require.Contains(t, downloadedURL, "https://generativelanguage.googleapis.com/download/v1beta/files/batch-out-2:download?alt=media")
	require.NotNil(t, usage2)
	require.Equal(t, 1400, usage2.InputTokens)
	require.Equal(t, 275, usage2.OutputTokens)
	require.Equal(t, 100, usage2.CacheReadInputTokens)

	boundAcc, boundUser, boundOK := svc.ResolveGeminiExtResource(context.Background(), "file", "batch-out-2")
	require.True(t, boundOK)
	require.Equal(t, acc.ID, boundAcc)
	require.Equal(t, int64(42), boundUser)

	// Non-terminal vs failed state
	_, pendingState, pendingTerm, err := svc.ExtractGeminiBatchUsage(context.Background(), acc, 42, []byte(`{"state":"BATCH_STATE_RUNNING"}`))
	require.NoError(t, err)
	require.Equal(t, "BATCH_STATE_RUNNING", pendingState)
	require.False(t, pendingTerm)

	failedUsage, failedState, failedTerm, err := svc.ExtractGeminiBatchUsage(context.Background(), acc, 42, []byte(`{"state":"BATCH_STATE_FAILED"}`))
	require.NoError(t, err)
	require.Equal(t, "BATCH_STATE_FAILED", failedState)
	require.True(t, failedTerm)
	require.Nil(t, failedUsage)
}

func TestGeminiLiveUsageTracker_MultiTurnAndResponseTokenCount(t *testing.T) {
	t.Parallel()

	tracker := &GeminiLiveUsageTracker{}
	// Turn 1 incremental + generationComplete + turnComplete (same turn must not double-count)
	tracker.ObserveServerFrame([]byte(`{"usageMetadata":{"promptTokenCount":100,"toolUsePromptTokenCount":20,"cachedContentTokenCount":10,"responseTokenCount":15,"thoughtsTokenCount":5}}`))
	tracker.ObserveServerFrame([]byte(`{"serverContent":{"generationComplete":true},"usageMetadata":{"promptTokenCount":100,"toolUsePromptTokenCount":20,"cachedContentTokenCount":10,"responseTokenCount":40,"thoughtsTokenCount":10}}`))
	tracker.ObserveServerFrame([]byte(`{"serverContent":{"turnComplete":true},"usageMetadata":{"promptTokenCount":100,"toolUsePromptTokenCount":20,"cachedContentTokenCount":10,"responseTokenCount":40,"thoughtsTokenCount":10}}`))
	// Turn 2 without explicit turnComplete before disconnect
	tracker.ObserveServerFrame([]byte(`{"usageMetadata":{"promptTokenCount":200,"responseTokenCount":30}}`))

	total := tracker.Total()
	require.Equal(t, 110+200, total.InputTokens)     // (100+20-10) + 200
	require.Equal(t, (40+10)+30, total.OutputTokens) // 50 + 30
	require.Equal(t, 10, total.CacheReadInputTokens)
}

func TestForwardGeminiAIStudioStream_ResumableUploadAndDownload(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	cache := &interactionTestCacheStub{}
	svc := &GeminiMessagesCompatService{
		cfg:   &config.Config{Security: config.SecurityConfig{URLAllowlist: config.URLAllowlistConfig{Enabled: false, AllowInsecureHTTP: true}}},
		cache: cache,
	}
	acc := &Account{
		ID:       77,
		Platform: PlatformGemini,
		Type:     AccountTypeAPIKey,
		Credentials: map[string]any{
			"api_key":  "AIza-up",
			"base_url": "https://generativelanguage.googleapis.com",
		},
	}

	var uploadedData []byte
	svc.httpUpstream = &extFuncHTTPUpstream{
		fn: func(req *http.Request) (*http.Response, error) {
			switch {
			case req.URL.Path == "/upload/v1beta/files" && req.URL.Query().Get("upload_id") == "":
				h := make(http.Header)
				h.Set("X-Goog-Upload-URL", "https://generativelanguage.googleapis.com/upload/v1beta/files?upload_id=goog-sess-99&upload_protocol=resumable")
				h.Set("X-Goog-Upload-Status", "active")
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     h,
					Body:       io.NopCloser(strings.NewReader("")),
				}, nil
			case req.URL.Path == "/upload/v1beta/files" && req.URL.Query().Get("upload_id") == "goog-sess-99":
				b, _ := io.ReadAll(req.Body)
				uploadedData = b
				return &http.Response{
					StatusCode: http.StatusOK,
					Header: http.Header{
						"Content-Type":         []string{"application/json"},
						"X-Goog-Upload-Status": []string{"final"},
					},
					Body: io.NopCloser(strings.NewReader(`{"file":{"name":"files/up-file-99"}}`)),
				}, nil
			default:
				payload := bytes.Repeat([]byte("A"), 128)
				return &http.Response{
					StatusCode:    http.StatusOK,
					ContentLength: int64(len(payload)),
					Header: http.Header{
						"Content-Type":        []string{"video/mp4"},
						"Accept-Ranges":       []string{"bytes"},
						"Content-Disposition": []string{`attachment; filename="clip.mp4"`},
					},
					Body: io.NopCloser(bytes.NewReader(payload)),
				}, nil
			}
		},
	}

	// Step 1: resumable start
	w1 := httptest.NewRecorder()
	c1, _ := gin.CreateTestContext(w1)
	c1.Request = httptest.NewRequest(http.MethodPost, "/upload/v1beta/files", strings.NewReader(`{"file":{"displayName":"clip"}}`))
	c1.Request.Host = "gateway.local:8080"
	c1.Request.Header.Set("X-Goog-Upload-Protocol", "resumable")
	c1.Request.Header.Set("X-Goog-Upload-Command", "start")

	require.NoError(t, svc.ForwardGeminiAIStudioStream(context.Background(), c1, acc, GeminiAIStudioStreamSpec{
		Method:   http.MethodPost,
		Path:     "/upload/v1beta/files",
		UserID:   1001,
		BindKind: "file",
	}))
	require.Equal(t, http.StatusOK, w1.Code)
	require.Equal(t, "active", w1.Header().Get("X-Goog-Upload-Status"))
	rewrittenURL := w1.Header().Get("X-Goog-Upload-URL")
	require.Contains(t, rewrittenURL, "http://gateway.local:8080/upload/v1beta/files?upload_id=")
	require.NotContains(t, rewrittenURL, "goog-sess-99")

	parsedRewritten, err := url.Parse(rewrittenURL)
	require.NoError(t, err)
	gwUploadID := parsedRewritten.Query().Get("upload_id")
	sess, ok := svc.GetGeminiUploadSession(context.Background(), gwUploadID)
	require.True(t, ok)
	require.Equal(t, int64(1001), sess.UserID)

	// Step 2: resumable upload finalize (>1MiB body streaming without full buffering)
	chunk := bytes.Repeat([]byte("Z"), (1<<20)+512)
	w2 := httptest.NewRecorder()
	c2, _ := gin.CreateTestContext(w2)
	c2.Request = httptest.NewRequest(http.MethodPut, "/upload/v1beta/files?upload_id="+gwUploadID, bytes.NewReader(chunk))
	c2.Request.Header.Set("X-Goog-Upload-Command", "upload, finalize")

	require.NoError(t, svc.ForwardGeminiAIStudioStream(context.Background(), c2, acc, GeminiAIStudioStreamSpec{
		Method:        http.MethodPut,
		Path:          "/upload/v1beta/files",
		UserID:        1001,
		BindKind:      "file",
		UploadSession: sess,
	}))
	require.Equal(t, http.StatusOK, w2.Code)
	require.Equal(t, "final", w2.Header().Get("X-Goog-Upload-Status"))
	require.Len(t, uploadedData, len(chunk))

	boundAcc, boundUser, found := svc.ResolveGeminiExtResource(context.Background(), "file", "up-file-99")
	require.True(t, found)
	require.Equal(t, acc.ID, boundAcc)
	require.Equal(t, int64(1001), boundUser)

	// Step 3: download streaming
	w3 := httptest.NewRecorder()
	c3, _ := gin.CreateTestContext(w3)
	c3.Request = httptest.NewRequest(http.MethodGet, "/download/v1beta/files/up-file-99:download?alt=media", nil)
	require.NoError(t, svc.ForwardGeminiAIStudioStream(context.Background(), c3, acc, GeminiAIStudioStreamSpec{
		Method:     http.MethodGet,
		Path:       "/download/v1beta/files/up-file-99:download",
		Query:      url.Values{"alt": {"media"}},
		UserID:     1001,
		IsDownload: true,
	}))
	require.Equal(t, http.StatusOK, w3.Code)
	require.Equal(t, "bytes", w3.Header().Get("Accept-Ranges"))
	require.Equal(t, 128, w3.Body.Len())
}

func TestForwardGeminiAIStudioStream_DoesNotTruncateLargeJSON(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)
	largeBody := append([]byte(`{"data":"`), bytes.Repeat([]byte("x"), geminiStreamJSONRespCap+512)...)
	largeBody = append(largeBody, []byte(`"}`)...)
	svc := &GeminiMessagesCompatService{
		cfg: &config.Config{Security: config.SecurityConfig{URLAllowlist: config.URLAllowlistConfig{Enabled: false, AllowInsecureHTTP: true}}},
		httpUpstream: &extFuncHTTPUpstream{fn: func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusCreated,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(bytes.NewReader(largeBody)),
			}, nil
		}},
	}
	account := &Account{ID: 9, Platform: PlatformGemini, Type: AccountTypeAPIKey,
		Credentials: map[string]any{"api_key": "test-key", "base_url": "https://generativelanguage.googleapis.com"}}
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/upload/v1beta/files", strings.NewReader(`{"file":{}}`))
	err := svc.ForwardGeminiAIStudioStream(context.Background(), c, account, GeminiAIStudioStreamSpec{
		Method: http.MethodPost, Path: "/upload/v1beta/files", UserID: 10, BindKind: "file",
	})
	require.NoError(t, err)
	require.Equal(t, http.StatusCreated, w.Code)
	require.Equal(t, largeBody, w.Body.Bytes())
}

func TestForwardGeminiAIStudioStream_FailoverPreservesSmallRequestBody(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	svc := &GeminiMessagesCompatService{
		cfg: &config.Config{Security: config.SecurityConfig{URLAllowlist: config.URLAllowlistConfig{Enabled: false, AllowInsecureHTTP: true}}},
		httpUpstream: &extFuncHTTPUpstream{
			fn: func(req *http.Request) (*http.Response, error) {
				return nil, context.DeadlineExceeded
			},
		},
	}
	acc := &Account{
		ID:          88,
		Platform:    PlatformGemini,
		Type:        AccountTypeAPIKey,
		Credentials: map[string]any{"api_key": "AIza-up", "base_url": "https://generativelanguage.googleapis.com"},
	}
	payload := `{"file":{"displayName":"small"}}`
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/upload/v1beta/files", strings.NewReader(payload))

	err := svc.ForwardGeminiAIStudioStream(context.Background(), c, acc, GeminiAIStudioStreamSpec{
		Method:   http.MethodPost,
		Path:     "/upload/v1beta/files",
		UserID:   1,
		BindKind: "file",
	})
	var failoverErr *UpstreamFailoverError
	require.ErrorAs(t, err, &failoverErr)

	remaining, readErr := io.ReadAll(c.Request.Body)
	require.NoError(t, readErr)
	require.Equal(t, payload, string(remaining))
}

func TestAccountStatsBaselineWithBatchCostFactor(t *testing.T) {
	t.Parallel()
	cs := newTestChannelServiceForStats(t, &Channel{
		ID:                         1,
		Status:                     StatusActive,
		GroupIDs:                   []int64{42},
		ApplyPricingToAccountStats: true,
	}, 42, PlatformGemini)
	usageLog := &UsageLog{}
	costFactor := 0.5
	totalCost := 1.0 // already scaled by 0.5 from $2.00 baseline
	statsBaselineCost := totalCost
	if costFactor > 0 && costFactor != 1 {
		statsBaselineCost /= costFactor
	}
	applyAccountStatsCost(context.Background(), usageLog, cs, nil, 7, 42, "gemini-3.5-flash-lite", "gemini-3.5-flash-lite",
		UsageTokens{InputTokens: 1_000_000, OutputTokens: 500_000}, statsBaselineCost, time.Now(), true)
	if costFactor > 0 && costFactor != 1 && usageLog.AccountStatsCost != nil {
		scaled := *usageLog.AccountStatsCost * costFactor
		usageLog.AccountStatsCost = &scaled
	}
	require.NotNil(t, usageLog.AccountStatsCost)
	require.InDelta(t, 1.0, *usageLog.AccountStatsCost, 1e-6)
}

func TestGeminiExtIDOnlyAgentResourceBindingAndFilter(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	svc := &GeminiMessagesCompatService{cache: &interactionTestCacheStub{}}

	// Managed Agents create returns an ID rather than a canonical name.
	svc.BindGeminiExtResourcesFromJSON(ctx, []byte("{\"id\":\"smoke-agent\",\"base_agent\":\"antigravity-preview-09-2026\"}"), "agent", 77, 42)
	accountID, ownerUserID, ok := svc.ResolveGeminiExtResource(ctx, "agent", "smoke-agent")
	require.True(t, ok)
	require.Equal(t, int64(77), accountID)
	require.Equal(t, int64(42), ownerUserID)

	rawList := []byte("{\"agents\":[{\"id\":\"smoke-agent\",\"base_agent\":\"antigravity-preview-09-2026\"},{\"id\":\"unowned-agent\"}],\"nextPageToken\":\"next\"}")
	filtered, err := svc.FilterGeminiExtListByUser(ctx, rawList, []string{"agents"}, "agent", 42)
	require.NoError(t, err)
	require.JSONEq(t,
		"{\"agents\":[{\"id\":\"smoke-agent\",\"base_agent\":\"antigravity-preview-09-2026\"}],\"nextPageToken\":\"next\"}",
		string(filtered),
	)

	otherUser, err := svc.FilterGeminiExtListByUser(ctx, rawList, []string{"agents"}, "agent", 999)
	require.NoError(t, err)
	require.JSONEq(t, "{\"agents\":[],\"nextPageToken\":\"next\"}", string(otherUser))

	// Malformed IDs must not create a resource binding.
	svc.BindGeminiExtResourcesFromJSON(ctx, []byte("{\"id\":\"../secret\"}"), "agent", 77, 42)
	_, _, ok = svc.ResolveGeminiExtResource(ctx, "agent", "../secret")
	require.False(t, ok)
}
