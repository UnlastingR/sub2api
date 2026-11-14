//go:build unit

package handler

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/service"
	coderws "github.com/coder/websocket"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestGeminiLiveWS_ConstrainedEphemeralTokenAndMultiTurnBilling(t *testing.T) {
	var (
		mu             sync.Mutex
		seenPath       string
		seenAccessTok  string
		seenSetupModel string
	)
	upstreamSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := coderws.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()

		mu.Lock()
		seenPath = r.URL.Path
		seenAccessTok = r.URL.Query().Get("access_token")
		mu.Unlock()

		// Read initial setup frame
		_, setupBytes, err := conn.Read(r.Context())
		if err != nil {
			return
		}
		mu.Lock()
		seenSetupModel = string(setupBytes)
		mu.Unlock()

		// Send setupComplete + Turn 1 usage (responseTokenCount + toolUsePromptTokenCount)
		_ = conn.Write(r.Context(), coderws.MessageText, []byte(`{"setupComplete":{}}`))
		_ = conn.Write(r.Context(), coderws.MessageText, []byte(`{"serverContent":{"turnComplete":true},"usageMetadata":{"promptTokenCount":100,"toolUsePromptTokenCount":50,"responseTokenCount":40,"thoughtsTokenCount":10}}`))
		// Send Turn 2 usage
		_ = conn.Write(r.Context(), coderws.MessageText, []byte(`{"serverContent":{"turnComplete":true},"usageMetadata":{"promptTokenCount":200,"responseTokenCount":60}}`))

		// Wait until client closes
		for {
			if _, _, err := conn.Read(r.Context()); err != nil {
				return
			}
		}
	}))
	defer upstreamSrv.Close()

	h, _, _, usageRepo, _, account, originKey, _ := newInteractionBillingTestHandler(t, `{}`)
	h.cfg.Security.URLAllowlist = config.URLAllowlistConfig{Enabled: false, AllowInsecureHTTP: true, AllowPrivateHosts: true}
	account.Credentials["base_url"] = upstreamSrv.URL
	account.Credentials["model_mapping"] = map[string]any{
		"gemini-3.5-flash-lite": "gemini-3.8-live",
	}

	rawEphemeral := "auth_tokens/eph-live-secret"
	require.NoError(t, h.geminiCompatService.SaveGeminiAuthTokenMeta(context.Background(), rawEphemeral, service.GeminiAuthTokenMeta{
		AccountID: account.ID,
		UserID:    originKey.UserID,
		APIKeyID:  originKey.ID,
		GroupID:   *originKey.GroupID,
		Model:     "gemini-3.5-flash-lite",
		ExpiresAt: time.Now().Add(10 * time.Minute).Unix(),
	}))

	router := gin.New()
	router.GET("/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained", func(c *gin.Context) {
		tok := ExtractGeminiLiveConstrainedAccessToken(c)
		meta, ok := h.ResolveGeminiAuthTokenMeta(c.Request.Context(), tok)
		require.True(t, ok)
		StashGeminiLiveConstrainedAuth(c, meta, tok)
		attachGeminiAuthContext(c, originKey, originKey.UserID)
		h.GeminiLiveWSConstrained(c)
	})
	gwSrv := httptest.NewServer(router)
	defer gwSrv.Close()

	wsURL := "ws" + strings.TrimPrefix(gwSrv.URL, "http") + "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained?access_token=" + rawEphemeral
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	clientConn, _, err := coderws.Dial(ctx, wsURL, nil)
	require.NoError(t, err)

	// Client sends setup with alias model
	require.NoError(t, clientConn.Write(ctx, coderws.MessageText, []byte(`{"setup":{"model":"models/gemini-3.5-flash-lite"}}`)))

	// Expect setupComplete + 2 usage server messages
	for i := 0; i < 3; i++ {
		_, msg, readErr := clientConn.Read(ctx)
		require.NoError(t, readErr)
		if i == 0 {
			require.Contains(t, string(msg), "setupComplete")
		}
	}
	require.NoError(t, clientConn.Close(coderws.StatusNormalClosure, "done"))

	mu.Lock()
	gotPath := seenPath
	gotTok := seenAccessTok
	gotSetup := seenSetupModel
	mu.Unlock()

	require.Equal(t, "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained", gotPath)
	require.Equal(t, rawEphemeral, gotTok)
	require.Contains(t, gotSetup, "models/gemini-3.8-live")

	// Wait briefly for server handler goroutine to tear down WS & submit usage log
	var logs []*service.UsageLog
	require.Eventually(t, func() bool {
		logs = usageRepo.snapshot()
		return len(logs) == 1
	}, 2*time.Second, 10*time.Millisecond)
	require.Len(t, logs, 1)
	require.Equal(t, 350, logs[0].InputTokens)  // (100+50) + 200
	require.Equal(t, 110, logs[0].OutputTokens) // (40+10) + 60
	require.Equal(t, originKey.ID, logs[0].APIKeyID)
}

func TestGeminiLiveWS_OpaqueUpstreamTokenAndAudioBilling(t *testing.T) {
	var (
		mu            sync.Mutex
		seenAccessTok string
	)
	upstreamSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := coderws.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.CloseNow() }()
		mu.Lock()
		seenAccessTok = r.URL.Query().Get("access_token")
		mu.Unlock()
		if _, _, err := conn.Read(r.Context()); err != nil {
			return
		}
		_ = conn.Write(r.Context(), coderws.MessageText, []byte(`{"setupComplete":{}}`))
		// 1M input audio tokens ($3.00) + 500k output audio tokens ($6.00) = $9.00 on gemini-3.8-live
		_ = conn.Write(r.Context(), coderws.MessageText, []byte(`{
			"serverContent":{"turnComplete":true},
			"usageMetadata":{
				"promptTokenCount":1000000,
				"responseTokenCount":500000,
				"promptTokensDetails":[{"modality":"AUDIO","tokenCount":1000000}],
				"responseTokensDetails":[{"modality":"AUDIO","tokenCount":500000}]
			}
		}`))
		for {
			if _, _, err := conn.Read(r.Context()); err != nil {
				return
			}
		}
	}))
	defer upstreamSrv.Close()

	h, _, _, usageRepo, _, account, originKey, _ := newInteractionBillingTestHandler(t, `{}`)
	h.cfg.Security.URLAllowlist = config.URLAllowlistConfig{Enabled: false, AllowInsecureHTTP: true, AllowPrivateHosts: true}
	account.Credentials["base_url"] = upstreamSrv.URL

	gwToken := "auth_tokens/gw-opaque-1"
	upToken := "auth_tokens/real-upstream-secret"
	require.NoError(t, h.geminiCompatService.SaveGeminiAuthTokenMeta(context.Background(), gwToken, service.GeminiAuthTokenMeta{
		AccountID:     account.ID,
		UserID:        originKey.UserID,
		APIKeyID:      originKey.ID,
		GroupID:       *originKey.GroupID,
		Model:         "gemini-3.8-live",
		UpstreamToken: upToken,
		ExpiresAt:     time.Now().Add(10 * time.Minute).Unix(),
	}))

	router := gin.New()
	router.GET("/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained", func(c *gin.Context) {
		tok := ExtractGeminiLiveConstrainedAccessToken(c)
		meta, ok := h.ResolveGeminiAuthTokenMeta(c.Request.Context(), tok)
		require.True(t, ok)
		StashGeminiLiveConstrainedAuth(c, meta, tok)
		attachGeminiAuthContext(c, originKey, originKey.UserID)
		h.GeminiLiveWSConstrained(c)
	})
	gwSrv := httptest.NewServer(router)
	defer gwSrv.Close()

	wsURL := "ws" + strings.TrimPrefix(gwSrv.URL, "http") + "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained?access_token=" + gwToken
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	clientConn, _, err := coderws.Dial(ctx, wsURL, nil)
	require.NoError(t, err)
	require.NoError(t, clientConn.Write(ctx, coderws.MessageText, []byte(`{"setup":{"model":"models/gemini-3.8-live"}}`)))
	for i := 0; i < 2; i++ {
		_, _, readErr := clientConn.Read(ctx)
		require.NoError(t, readErr)
	}
	require.NoError(t, clientConn.Close(coderws.StatusNormalClosure, "done"))

	mu.Lock()
	gotTok := seenAccessTok
	mu.Unlock()
	require.Equal(t, upToken, gotTok)

	var logs []*service.UsageLog
	require.Eventually(t, func() bool {
		logs = usageRepo.snapshot()
		return len(logs) == 1
	}, 2*time.Second, 10*time.Millisecond)
	require.InDelta(t, 9.0, logs[0].TotalCost, 1e-6)
}
