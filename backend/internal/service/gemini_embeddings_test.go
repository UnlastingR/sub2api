//go:build unit

package service

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func newGeminiEmbeddingTestService(status int, body string) (*GeminiMessagesCompatService, *geminiCompatHTTPUpstreamStub) {
	stub := &geminiCompatHTTPUpstreamStub{response: &http.Response{
		StatusCode: status,
		Header:     http.Header{"Content-Type": {"application/json"}, "X-Request-Id": {"embedding-request"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}}
	return &GeminiMessagesCompatService{cfg: &config.Config{}, httpUpstream: stub}, stub
}

func embeddingTestContext(body string) (*gin.Context, *httptest.ResponseRecorder) {
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(body))
	return c, rec
}

func TestGeminiEmbeddingsNativeActionsPreserveBodyAndUsage(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, tc := range []struct{ action, request, response string }{
		{"embedContent", `{"content":{"parts":[{"text":"hello"}]},"outputDimensionality":2}`, `{"embedding":{"values":[0.25,-0.5]},"usageMetadata":{"promptTokenCount":4}}`},
		{"batchEmbedContents", `{"requests":[{"model":"models/gemini-embedding-2","content":{"parts":[{"text":"hello"}]}},{"model":"models/gemini-embedding-2","content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"test"}}]}}]}`, `{"embeddings":[{"values":[0.25,-0.5]},{"values":[0.75,1]}],"usageMetadata":{"promptTokenCount":9,"promptTokenDetails":[{"modality":"TEXT","tokenCount":4},{"modality":"IMAGE","tokenCount":5}]}}`},
	} {
		t.Run(tc.action, func(t *testing.T) {
			svc, stub := newGeminiEmbeddingTestService(http.StatusOK, tc.response)
			c, rec := embeddingTestContext(tc.request)
			result, err := svc.ForwardNative(context.Background(), c, geminiSignalTestAccount(), "gemini-embedding-2", tc.action, false, []byte(tc.request))
			require.NoError(t, err)
			require.Equal(t, http.StatusOK, rec.Code)
			require.JSONEq(t, tc.response, rec.Body.String())
			require.Positive(t, result.Usage.InputTokens)
			require.Zero(t, result.Usage.OutputTokens)
			require.Empty(t, GetOpsStreamErrors(c), "embedding bodies must not be marked as empty generation responses")
			require.Equal(t, "/v1beta/models/gemini-embedding-2:"+tc.action, stub.lastReq.URL.Path)
			require.Equal(t, "test-key", stub.lastReq.Header.Get("x-goog-api-key"))
			sent, err := io.ReadAll(stub.lastReq.Body)
			require.NoError(t, err)
			require.JSONEq(t, tc.request, string(sent), "native media and optional fields must be preserved")
		})
	}
}

func TestGeminiEmbeddingsOpenAIAdapterBatchBase64AndModelMapping(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := `{"model":"embedding-alias","input":["first","second"],"dimensions":2,"encoding_format":"base64"}`
	c, rec := embeddingTestContext(body)
	model, nativeBody, err := PrepareGeminiEmbeddingsRequest(c, []byte(body))
	require.NoError(t, err)
	require.Equal(t, []byte(body), GeminiEmbeddingsOriginalBody(c))
	svc, stub := newGeminiEmbeddingTestService(http.StatusOK, `{"embeddings":[{"values":[0.25,-0.5]},{"values":[0.75,1]}],"tokenCount":"7"}`)
	account := geminiSignalTestAccount()
	account.Credentials["model_mapping"] = map[string]any{"embedding-alias": "gemini-embedding-2"}
	result, err := svc.ForwardNative(context.Background(), c, account, model, "batchEmbedContents", false, nativeBody)
	require.NoError(t, err)
	require.Equal(t, 7, result.Usage.InputTokens)
	require.Zero(t, result.Usage.OutputTokens)
	require.Equal(t, "gemini-embedding-2", result.UpstreamModel)
	require.Equal(t, "/v1beta/models/gemini-embedding-2:batchEmbedContents", stub.lastReq.URL.Path)
	sent, err := io.ReadAll(stub.lastReq.Body)
	require.NoError(t, err)
	require.Equal(t, "models/gemini-embedding-2", gjson.GetBytes(sent, "requests.0.model").String())
	require.Equal(t, "models/gemini-embedding-2", gjson.GetBytes(sent, "requests.1.model").String())
	require.Equal(t, int64(2), gjson.GetBytes(sent, "requests.0.outputDimensionality").Int())
	require.Equal(t, "second", gjson.GetBytes(sent, "requests.1.content.parts.0.text").String())
	got := gjson.Parse(rec.Body.String())
	require.Equal(t, "list", got.Get("object").String())
	require.Equal(t, "embedding-alias", got.Get("model").String())
	require.Equal(t, int64(7), got.Get("usage.prompt_tokens").Int())
	require.Equal(t, int64(7), got.Get("usage.total_tokens").Int())
	require.Equal(t, int64(1), got.Get("data.1.index").Int())
	encoded := got.Get("data.0.embedding").String()
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	require.NoError(t, err)
	require.Len(t, decoded, 8)
	require.Equal(t, float32(0.25), math.Float32frombits(binary.LittleEndian.Uint32(decoded)))
	require.Equal(t, float32(-0.5), math.Float32frombits(binary.LittleEndian.Uint32(decoded[4:])))
}

func TestGeminiEmbeddingsOpenAIAdapterSingleFloat(t *testing.T) {
	body := `{"model":"models/gemini-embedding-2","input":"hello"}`
	c, rec := embeddingTestContext(body)
	model, nativeBody, err := PrepareGeminiEmbeddingsRequest(c, []byte(body))
	require.NoError(t, err)
	svc, _ := newGeminiEmbeddingTestService(http.StatusOK, `{"embeddings":[{"values":[0.25,-0.5]}],"usageMetadata":{"promptTokenCount":4}}`)
	result, err := svc.ForwardNative(context.Background(), c, geminiSignalTestAccount(), model, "batchEmbedContents", false, nativeBody)
	require.NoError(t, err)
	require.Equal(t, 4, result.Usage.InputTokens)
	require.Equal(t, 0.25, gjson.Get(rec.Body.String(), "data.0.embedding.0").Float())
	require.Equal(t, "hello", gjson.GetBytes(nativeBody, "requests.0.content.parts.0.text").String())
}

func TestGeminiEmbeddingsValidationRejectsUnsupportedInputs(t *testing.T) {
	for _, body := range []string{
		`{}`, `[]`, `null`, `{invalid`,
		`{"model":"../escape","input":"hello"}`,
		`{"model":"gemini-embedding-2","input":null}`,
		`{"model":"gemini-embedding-2","input":[]}`,
		`{"model":"gemini-embedding-2","input":[1,2]}`,
		`{"model":"gemini-embedding-2","input":["hello",3]}`,
		`{"model":"gemini-embedding-2","input":""}`,
		`{"model":"gemini-embedding-2","input":"hello","dimensions":0}`,
		`{"model":"gemini-embedding-2","input":"hello","dimensions":1.5}`,
		`{"model":"gemini-embedding-2","input":"hello","encoding_format":"binary"}`,
		`{"model":"gemini-embedding-2","input":"hello","stream":true}`,
	} {
		t.Run(body, func(t *testing.T) {
			c, _ := embeddingTestContext(body)
			_, _, err := PrepareGeminiEmbeddingsRequest(c, []byte(body))
			require.Error(t, err)
			require.Nil(t, GeminiEmbeddingsOriginalBody(c))
		})
	}
}

func TestGeminiEmbeddingsMalformedUpstreamDoesNotProduceBillableResult(t *testing.T) {
	for _, response := range []string{
		`{}`, `{"embeddings":[]}`, `{"embeddings":[{"values":[]}]}`,
		`{"embeddings":[{"values":[null]}]}`, `{"embeddings":[{"values":["invalid"]}]}`,
		`{"embeddings":[{"values":[1e100]}]}`, `{"embeddings":[{"values":[1]},{"values":[2]}]}`,
	} {
		t.Run(response, func(t *testing.T) {
			body := `{"model":"gemini-embedding-2","input":"hello"}`
			c, rec := embeddingTestContext(body)
			model, nativeBody, err := PrepareGeminiEmbeddingsRequest(c, []byte(body))
			require.NoError(t, err)
			svc, _ := newGeminiEmbeddingTestService(http.StatusOK, response)
			result, err := svc.ForwardNative(context.Background(), c, geminiSignalTestAccount(), model, "batchEmbedContents", false, nativeBody)
			require.Error(t, err)
			require.Nil(t, result)
			require.Equal(t, http.StatusBadGateway, rec.Code)
		})
	}
}

func TestGeminiEmbeddingsNativeSingleAccountMappingPreservesOtherFields(t *testing.T) {
	body := []byte(`{"model":"models/embedding-alias","content":{"parts":[{"text":"hello"}]},"outputDimensionality":128,"taskType":"RETRIEVAL_QUERY"}`)
	mapped, err := mapGeminiEmbeddingRequestModels(body, "gemini-embedding-2", "embedContent")
	require.NoError(t, err)
	require.Equal(t, "models/gemini-embedding-2", gjson.GetBytes(mapped, "model").String())
	require.Equal(t, "RETRIEVAL_QUERY", gjson.GetBytes(mapped, "taskType").String())
	require.Equal(t, int64(128), gjson.GetBytes(mapped, "outputDimensionality").Int())
	require.True(t, bytes.Contains(mapped, []byte("hello")))
}

func TestGeminiEmbeddingsCompositeMappingUpdatesBatchModels(t *testing.T) {
	body := `{"model":"embedding-alias","input":"hello"}`
	c, rec := embeddingTestContext(body)
	_, nativeBody, err := PrepareGeminiEmbeddingsRequest(c, []byte(body))
	require.NoError(t, err)
	ctx := WithCompositeRouteDecision(context.Background(), CompositeRouteDecision{
		Matched: true, TargetPlatform: PlatformGemini, PublicModel: "embedding-alias", UpstreamModel: "gemini-embedding-2",
	})
	svc, stub := newGeminiEmbeddingTestService(http.StatusOK, `{"embeddings":[{"values":[0.25]}],"usageMetadata":{"promptTokenCount":4}}`)
	_, err = svc.ForwardNative(ctx, c, geminiSignalTestAccount(), "gemini-embedding-2", "batchEmbedContents", false, nativeBody)
	require.NoError(t, err)
	sent, err := io.ReadAll(stub.lastReq.Body)
	require.NoError(t, err)
	require.Equal(t, "models/gemini-embedding-2", gjson.GetBytes(sent, "requests.0.model").String())
	require.Equal(t, "embedding-alias", gjson.Get(rec.Body.String(), "model").String())
}
