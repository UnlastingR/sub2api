//go:build unit

package service

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

type vertexEmbeddingTestResponse struct {
	status int
	body   string
	err    error
}

type vertexEmbeddingTestUpstream struct {
	responses []vertexEmbeddingTestResponse
	requests  []*http.Request
	bodies    []string
}

type vertexEmbeddingExpiringTokenCache struct {
	fakeGeminiTokenCache
	reads int
}

func (c *vertexEmbeddingExpiringTokenCache) GetAccessToken(context.Context, string) (string, error) {
	c.reads++
	if c.reads == 1 {
		return "cached-vertex-token", nil
	}
	return "", errors.New("expired")
}

func (c *vertexEmbeddingExpiringTokenCache) AcquireRefreshLock(context.Context, string, time.Duration) (bool, error) {
	return true, nil
}

func (s *vertexEmbeddingTestUpstream) Do(req *http.Request, _ string, _ int64, _ int) (*http.Response, error) {
	body, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	s.requests = append(s.requests, req)
	s.bodies = append(s.bodies, string(body))
	if len(s.requests) > len(s.responses) {
		return nil, errors.New("unexpected upstream request")
	}
	response := s.responses[len(s.requests)-1]
	if response.err != nil {
		return nil, response.err
	}
	return &http.Response{
		StatusCode: response.status,
		Header:     http.Header{"Content-Type": {"application/json"}, "X-Request-Id": {"vertex-embedding-test"}},
		Body:       io.NopCloser(strings.NewReader(response.body)),
	}, nil
}

func (s *vertexEmbeddingTestUpstream) DoWithTLS(req *http.Request, proxyURL string, accountID int64, concurrency int, _ *tlsfingerprint.Profile) (*http.Response, error) {
	return s.Do(req, proxyURL, accountID, concurrency)
}

func vertexEmbeddingTestService(responses ...vertexEmbeddingTestResponse) (*GeminiMessagesCompatService, *vertexEmbeddingTestUpstream) {
	upstream := &vertexEmbeddingTestUpstream{responses: responses}
	return &GeminiMessagesCompatService{
		cfg:           &config.Config{},
		httpUpstream:  upstream,
		tokenProvider: NewGeminiTokenProvider(nil, &fakeGeminiTokenCache{token: "cached-vertex-token"}, nil),
	}, upstream
}

func TestVertexGeminiEmbedContentServiceAccount(t *testing.T) {
	body := `{"model":"models/gemini-embedding-2","content":{"parts":[{"text":"hello"}]},"outputDimensionality":2,"embedContentConfig":{"documentOcr":true}}`
	response := `{"embedding":{"values":[0.25,-0.5]},"usageMetadata":{"promptTokenCount":4}}`
	svc, upstream := vertexEmbeddingTestService(vertexEmbeddingTestResponse{status: http.StatusOK, body: response})
	account := newVertexTokenTestAccount(t)
	c, rec := embeddingTestContext(body)

	result, err := svc.ForwardNative(context.Background(), c, account, "gemini-embedding-2", "embedContent", false, []byte(body))
	require.NoError(t, err)
	require.Equal(t, 4, result.Usage.InputTokens)
	require.JSONEq(t, response, rec.Body.String())
	require.Len(t, upstream.requests, 1)
	require.Equal(t, "https://aiplatform.googleapis.com/v1/projects/proj-1/locations/global/publishers/google/models/gemini-embedding-2:embedContent", upstream.requests[0].URL.String())
	require.Equal(t, "Bearer cached-vertex-token", upstream.requests[0].Header.Get("Authorization"))
	require.JSONEq(t, `{"content":{"parts":[{"text":"hello"}]},"outputDimensionality":2,"embedContentConfig":{"documentOcr":true}}`, upstream.bodies[0])
}

func TestVertexGeminiBatchEmbedContentsPreservesOrderAndUsage(t *testing.T) {
	body := `{"requests":[{"model":"models/embedding-alias","content":{"parts":[{"text":"one"}]},"outputDimensionality":2},{"model":"models/embedding-alias","content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"test"}}]},"outputDimensionality":2}]}`
	svc, upstream := vertexEmbeddingTestService(
		vertexEmbeddingTestResponse{status: http.StatusOK, body: `{"embedding":{"values":[0.25,-0.5]},"usageMetadata":{"promptTokenCount":4,"promptTokensDetails":[{"modality":"TEXT","tokenCount":4}]}}`},
		vertexEmbeddingTestResponse{status: http.StatusOK, body: `{"embedding":{"values":[0.75,1]},"usageMetadata":{"promptTokenCount":5,"promptTokensDetails":[{"modality":"IMAGE","tokenCount":5}]}}`},
	)
	account := geminiVertexAPIKeyTestAccount("my-project")
	account.Credentials["model_mapping"] = map[string]any{"embedding-alias": "gemini-embedding-2"}
	c, rec := embeddingTestContext(body)

	result, err := svc.ForwardNative(context.Background(), c, account, "embedding-alias", "batchEmbedContents", false, []byte(body))
	require.NoError(t, err)
	require.Equal(t, "gemini-embedding-2", result.UpstreamModel)
	require.Equal(t, 9, result.Usage.InputTokens)
	require.Len(t, upstream.requests, 2)
	for _, request := range upstream.requests {
		require.Equal(t, "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-embedding-2:embedContent", request.URL.String())
		requireVertexAPIKeyAuth(t, request)
	}
	require.JSONEq(t, `{"content":{"parts":[{"text":"one"}]},"outputDimensionality":2}`, upstream.bodies[0])
	require.JSONEq(t, `{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"test"}}]},"outputDimensionality":2}`, upstream.bodies[1])
	require.JSONEq(t, `{"embeddings":[{"values":[0.25,-0.5]},{"values":[0.75,1]}],"usageMetadata":{"promptTokenCount":9,"promptTokenDetails":[{"modality":"TEXT","tokenCount":4},{"modality":"IMAGE","tokenCount":5}]}}`, rec.Body.String())
}

func TestVertexGeminiOpenAIEmbeddingsBatch(t *testing.T) {
	body := `{"model":"gemini-embedding-2","input":["first","second"],"dimensions":2,"encoding_format":"base64"}`
	c, rec := embeddingTestContext(body)
	model, nativeBody, err := PrepareGeminiEmbeddingsRequest(c, []byte(body))
	require.NoError(t, err)
	svc, upstream := vertexEmbeddingTestService(
		vertexEmbeddingTestResponse{status: http.StatusOK, body: `{"embedding":{"values":[0.25,-0.5]},"usageMetadata":{"promptTokenCount":4}}`},
		vertexEmbeddingTestResponse{status: http.StatusOK, body: `{"embedding":{"values":[0.75,1]},"usageMetadata":{"promptTokenCount":5}}`},
	)

	result, err := svc.ForwardNative(context.Background(), c, geminiVertexAPIKeyTestAccount(""), model, "batchEmbedContents", false, nativeBody)
	require.NoError(t, err)
	require.Equal(t, 9, result.Usage.InputTokens)
	require.Len(t, upstream.requests, 2)
	require.Equal(t, "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-embedding-2:embedContent", upstream.requests[0].URL.String())
	require.Equal(t, "first", gjson.Get(upstream.bodies[0], "content.parts.0.text").String())
	require.Equal(t, "second", gjson.Get(upstream.bodies[1], "content.parts.0.text").String())
	require.Equal(t, int64(2), gjson.Get(upstream.bodies[1], "outputDimensionality").Int())
	require.Equal(t, "list", gjson.Get(rec.Body.String(), "object").String())
	require.Equal(t, "AACAPgAAAL8=", gjson.Get(rec.Body.String(), "data.0.embedding").String())
	require.Equal(t, int64(1), gjson.Get(rec.Body.String(), "data.1.index").Int())
	require.Equal(t, int64(9), gjson.Get(rec.Body.String(), "usage.prompt_tokens").Int())
}

func TestVertexGeminiEmbeddingBatchValidatesBeforeSending(t *testing.T) {
	valid := `{"model":"models/gemini-embedding-2","content":{"parts":[{"text":"ok"}]}}`
	for name, body := range map[string]string{
		"empty":          `{"requests":[]}`,
		"missing":        `{"requests":[{}]}`,
		"invalid part":   `{"requests":[{"content":{"parts":[null]}}]}`,
		"mismatched":     fmt.Sprintf(`{"requests":[%s,{"model":"models/other","content":{"parts":[{"text":"second"}]}}]}`, valid),
		"over limit":     `{"requests":[` + strings.Repeat(valid+",", maxVertexGeminiEmbeddingBatch) + valid + `]}`,
		"malformed json": `{"requests":[`,
	} {
		t.Run(name, func(t *testing.T) {
			svc, upstream := vertexEmbeddingTestService()
			c, rec := embeddingTestContext(body)
			result, err := svc.ForwardNative(context.Background(), c, geminiVertexAPIKeyTestAccount(""), "gemini-embedding-2", "batchEmbedContents", false, []byte(body))
			require.Error(t, err)
			require.Nil(t, result)
			require.Equal(t, http.StatusBadRequest, rec.Code)
			require.Empty(t, upstream.requests)
		})
	}
}

func TestVertexGeminiEmbeddingRetriesOnlyFailedItem(t *testing.T) {
	body := `{"requests":[{"content":{"parts":[{"text":"first"}]}},{"content":{"parts":[{"text":"second"}]}}]}`
	svc, upstream := vertexEmbeddingTestService(
		vertexEmbeddingTestResponse{status: http.StatusOK, body: `{"embedding":{"values":[1]},"usageMetadata":{"promptTokenCount":2}}`},
		vertexEmbeddingTestResponse{status: http.StatusServiceUnavailable, body: `{"error":{"message":"unavailable"}}`},
		vertexEmbeddingTestResponse{status: http.StatusOK, body: `{"embedding":{"values":[2]},"usageMetadata":{"promptTokenCount":3}}`},
	)
	svc.cfg.Gateway.MaxSameAccountRetriesGemini = 1
	c, rec := embeddingTestContext(body)

	result, err := svc.ForwardNative(context.Background(), c, geminiVertexAPIKeyTestAccount(""), "gemini-embedding-2", "batchEmbedContents", false, []byte(body))
	require.NoError(t, err)
	require.Equal(t, 5, result.Usage.InputTokens)
	require.Len(t, upstream.requests, 3)
	require.Equal(t, "first", gjson.Get(upstream.bodies[0], "content.parts.0.text").String())
	require.Equal(t, "second", gjson.Get(upstream.bodies[1], "content.parts.0.text").String())
	require.JSONEq(t, upstream.bodies[1], upstream.bodies[2])
	require.Equal(t, int64(2), gjson.Get(rec.Body.String(), "embeddings.1.values.0").Int())
}

func TestVertexGeminiEmbeddingPartialFailureDoesNotFailOver(t *testing.T) {
	body := `{"requests":[{"content":{"parts":[{"text":"first"}]}},{"content":{"parts":[{"text":"second"}]}}]}`
	for _, tc := range []struct {
		name     string
		response vertexEmbeddingTestResponse
		status   int
	}{
		{"upstream", vertexEmbeddingTestResponse{status: http.StatusServiceUnavailable, body: `{"error":{"message":"unavailable"}}`}, http.StatusServiceUnavailable},
		{"transport", vertexEmbeddingTestResponse{err: errors.New("connection lost")}, http.StatusBadGateway},
		{"malformed", vertexEmbeddingTestResponse{status: http.StatusOK, body: `{"embedding":{"values":[null]}}`}, http.StatusBadGateway},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc, upstream := vertexEmbeddingTestService(
				vertexEmbeddingTestResponse{status: http.StatusOK, body: `{"embedding":{"values":[1]},"usageMetadata":{"promptTokenCount":2}}`},
				tc.response,
			)
			c, rec := embeddingTestContext(body)
			result, err := svc.ForwardNative(context.Background(), c, geminiVertexAPIKeyTestAccount(""), "gemini-embedding-2", "batchEmbedContents", false, []byte(body))
			require.Error(t, err)
			require.Nil(t, result)
			var failover *UpstreamFailoverError
			require.False(t, errors.As(err, &failover))
			require.Len(t, upstream.requests, 2)
			require.Equal(t, tc.status, rec.Code)
		})
	}
}

func TestVertexGeminiEmbeddingPartialAuthFailureDisablesAccountWithoutFailover(t *testing.T) {
	useVertexTokenEndpoint(t, http.StatusBadRequest, vertexAccountNotFoundTokenBody)
	body := `{"requests":[{"content":{"parts":[{"text":"first"}]}},{"content":{"parts":[{"text":"second"}]}}]}`
	svc, upstream := vertexEmbeddingTestService(vertexEmbeddingTestResponse{status: http.StatusOK, body: `{"embedding":{"values":[1]},"usageMetadata":{"promptTokenCount":2}}`})
	repo := &errorPolicyRepoStub{}
	svc.accountRepo = repo
	svc.tokenProvider = NewGeminiTokenProvider(repo, &vertexEmbeddingExpiringTokenCache{}, nil)
	c, rec := embeddingTestContext(body)

	result, err := svc.ForwardNative(context.Background(), c, newVertexTokenTestAccount(t), "gemini-embedding-2", "batchEmbedContents", false, []byte(body))
	require.Error(t, err)
	require.Nil(t, result)
	var failover *UpstreamFailoverError
	require.False(t, errors.As(err, &failover))
	require.Len(t, upstream.requests, 1)
	require.Equal(t, 1, repo.setErrCalls)
	require.Equal(t, http.StatusBadGateway, rec.Code)
}
