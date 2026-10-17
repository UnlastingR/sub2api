package handler

import (
	"bytes"
	"io"
	"net/http"

	pkghttputil "github.com/Wei-Shaw/sub2api/internal/pkg/httputil"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
)

// GeminiEmbeddings adapts the OpenAI wire format and uses the existing native
// Gemini gateway for authentication, audit, scheduling, failover and billing.
func (h *GatewayHandler) GeminiEmbeddings(c *gin.Context) {
	body, err := pkghttputil.ReadRequestBodyWithPrealloc(c.Request)
	if err != nil {
		if maxErr, ok := extractMaxBytesError(err); ok {
			h.chatCompletionsErrorResponse(c, http.StatusRequestEntityTooLarge, "invalid_request_error", buildBodyTooLargeMessage(maxErr.Limit))
			return
		}
		h.chatCompletionsErrorResponse(c, http.StatusBadRequest, "invalid_request_error", "Failed to read request body")
		return
	}
	model, nativeBody, err := service.PrepareGeminiEmbeddingsRequest(c, body)
	if err != nil {
		h.chatCompletionsErrorResponse(c, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	originalBody, originalLength, originalParams := c.Request.Body, c.Request.ContentLength, c.Params
	defer func() {
		c.Request.Body, c.Request.ContentLength, c.Params = originalBody, originalLength, originalParams
	}()
	c.Request.Body = io.NopCloser(bytes.NewReader(nativeBody))
	c.Request.ContentLength = int64(len(nativeBody))
	c.Params = append(c.Params, gin.Param{Key: "modelAction", Value: "/" + model + ":batchEmbedContents"})
	h.GeminiV1BetaModels(c)
}
