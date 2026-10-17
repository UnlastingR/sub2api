package handler

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestGeminiEmbeddingsRejectsInvalidInputBeforeScheduling(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{"model":"gemini-embedding-2","input":[1,2,3]}`))
	(&GatewayHandler{}).GeminiEmbeddings(c)
	require.Equal(t, http.StatusBadRequest, rec.Code)
	require.Contains(t, rec.Body.String(), "input must be a string")
}

func TestGeminiEmbeddingsKeepsNativeAuthenticationGateAndRestoresParams(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(`{"model":"gemini-embedding-2","input":"hello"}`))
	(&GatewayHandler{}).GeminiEmbeddings(c)
	require.Equal(t, http.StatusUnauthorized, rec.Code)
	require.Contains(t, rec.Body.String(), "Invalid API key")
	require.Empty(t, c.Params)
}
