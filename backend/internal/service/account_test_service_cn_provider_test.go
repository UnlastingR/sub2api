//go:build unit

package service

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestAccountTestService_CNProvidersDefaultToChatCompletions(t *testing.T) {
	platforms := []string{PlatformKimi, PlatformZhipu, PlatformDeepseek}
	for i, platform := range platforms {
		t.Run(platform, func(t *testing.T) {
			ctx, recorder := newTestContext()
			upstreamBody := strings.Join([]string{
				`data: {"id":"chatcmpl_test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"pong"},"finish_reason":null}]}`,
				"",
				`data: {"id":"chatcmpl_test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
				"",
				"data: [DONE]",
				"",
			}, "\n")
			upstream := &httpUpstreamRecorder{resp: &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       io.NopCloser(strings.NewReader(upstreamBody)),
			}}
			account := &Account{
				ID:          int64(600 + i),
				Platform:    platform,
				Type:        AccountTypeAPIKey,
				Concurrency: 1,
				Credentials: map[string]any{
					"api_key":  "sk-cn-test",
					"base_url": "https://compat-upstream.example/v1",
				},
			}
			repo := &mockAccountRepoForGemini{accountsByID: map[int64]*Account{account.ID: account}}
			svc := &AccountTestService{
				accountRepo:  repo,
				httpUpstream: upstream,
				cfg:          &config.Config{Security: config.SecurityConfig{URLAllowlist: config.URLAllowlistConfig{Enabled: false}}},
			}

			err := svc.TestAccountConnection(ctx, account.ID, "DeepSeek-V4-Flash", "hi", "")
			require.NoError(t, err)
			require.NotNil(t, upstream.lastReq)
			require.Equal(t, "https://compat-upstream.example/v1/chat/completions", upstream.lastReq.URL.String())
			require.Equal(t, "Bearer sk-cn-test", upstream.lastReq.Header.Get("Authorization"))
			require.Empty(t, upstream.lastReq.Header.Get("anthropic-version"))
			require.Equal(t, "DeepSeek-V4-Flash", gjson.GetBytes(upstream.lastBody, "model").String())
			require.Equal(t, "hi", gjson.GetBytes(upstream.lastBody, "messages.0.content").String())
			require.Contains(t, recorder.Body.String(), "已通过 /v1/chat/completions 验证")
			require.Contains(t, recorder.Body.String(), `"success":true`)
		})
	}
}
