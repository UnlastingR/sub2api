//go:build unit

package handler

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func attachGeminiAuthContext(c *gin.Context, apiKey *service.APIKey, userID int64) {
	c.Set(string(middleware.ContextKeyAPIKey), apiKey)
	c.Set(string(middleware.ContextKeyUser), middleware.AuthSubject{
		UserID:      userID,
		Concurrency: 5,
	})
}

func TestGeminiExtEndpoints_TenantIsolationAndManagedAgents(t *testing.T) {
	h, _, _, _, upstream, _, originKey, _ := newInteractionBillingTestHandler(t, `{"name":"agents/agent-1","displayName":"MyAgent"}`)

	// User 10 creates an agent
	rec1 := httptest.NewRecorder()
	c1, _ := gin.CreateTestContext(rec1)
	c1.Request = httptest.NewRequest(http.MethodPost, "/v1beta/agents", strings.NewReader(`{"displayName":"MyAgent","agentConfig":{"model":"gemini-3.5-flash-lite"}}`))
	attachGeminiAuthContext(c1, originKey, originKey.UserID)

	h.GeminiV1BetaAgents(c1)
	require.Equal(t, http.StatusOK, rec1.Code)
	require.Contains(t, rec1.Body.String(), "agents/agent-1")
	require.Equal(t, "/v1beta/agents", upstream.lastReq.URL.Path)

	// Same user can GET it
	rec2 := httptest.NewRecorder()
	c2, _ := gin.CreateTestContext(rec2)
	c2.Request = httptest.NewRequest(http.MethodGet, "/v1beta/agents/agent-1", nil)
	c2.Params = gin.Params{{Key: "agentID", Value: "agent-1"}}
	attachGeminiAuthContext(c2, originKey, originKey.UserID)

	h.GeminiV1BetaAgentByID(c2)
	require.Equal(t, http.StatusOK, rec2.Code)

	// Different user gets 404 (tenant isolation)
	otherUser := &service.User{ID: 999, Balance: 50}
	otherKey := &service.APIKey{ID: 303, UserID: 999, User: otherUser, GroupID: originKey.GroupID, Group: originKey.Group}
	rec3 := httptest.NewRecorder()
	c3, _ := gin.CreateTestContext(rec3)
	c3.Request = httptest.NewRequest(http.MethodGet, "/v1beta/agents/agent-1", nil)
	c3.Params = gin.Params{{Key: "agentID", Value: "agent-1"}}
	attachGeminiAuthContext(c3, otherKey, otherUser.ID)

	h.GeminiV1BetaAgentByID(c3)
	require.Equal(t, http.StatusNotFound, rec3.Code)
}

func TestGeminiExtEndpoints_WebhookActionAndMethodChecks(t *testing.T) {
	h, _, _, _, upstream, account, originKey, _ := newInteractionBillingTestHandler(t, `{"secret":"rot-1"}`)
	h.geminiCompatService.BindGeminiExtResourceAccount(context.Background(), "webhook", "wh-1", account.ID, originKey.UserID, 0)

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1beta/webhooks/wh-1:rotateSigningSecret", strings.NewReader(`{}`))
	c.Params = gin.Params{{Key: "webhookID", Value: "wh-1:rotateSigningSecret"}}
	attachGeminiAuthContext(c, originKey, originKey.UserID)

	h.GeminiV1BetaWebhookByID(c)
	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, "/v1beta/webhooks/wh-1:rotateSigningSecret", upstream.lastReq.URL.Path)

	// Unowned single-resource webhook fails closed with 404
	recUnowned := httptest.NewRecorder()
	cUnowned, _ := gin.CreateTestContext(recUnowned)
	cUnowned.Request = httptest.NewRequest(http.MethodPost, "/v1beta/webhooks/wh-missing:rotateSigningSecret", strings.NewReader(`{}`))
	cUnowned.Params = gin.Params{{Key: "webhookID", Value: "wh-missing:rotateSigningSecret"}}
	attachGeminiAuthContext(cUnowned, originKey, originKey.UserID)

	h.GeminiV1BetaWebhookByID(cUnowned)
	require.Equal(t, http.StatusNotFound, recUnowned.Code)

	// GET on POST-only action rejected
	recBad := httptest.NewRecorder()
	cBad, _ := gin.CreateTestContext(recBad)
	cBad.Request = httptest.NewRequest(http.MethodGet, "/v1beta/webhooks/wh-1:ping", nil)
	cBad.Params = gin.Params{{Key: "webhookID", Value: "wh-1:ping"}}
	attachGeminiAuthContext(cBad, originKey, originKey.UserID)

	h.GeminiV1BetaWebhookByID(cBad)
	require.Equal(t, http.StatusMethodNotAllowed, recBad.Code)
}

func TestGeminiExtEndpoints_RejectsConflictingResourceAccounts(t *testing.T) {
	h, _, _, _, _, account, originKey, _ := newInteractionBillingTestHandler(t, `{}`)
	h.geminiCompatService.BindGeminiExtResourceAccount(context.Background(), "store", "store-1", account.ID, originKey.UserID, 0)
	h.geminiCompatService.BindGeminiExtResourceAccount(context.Background(), "file", "file-1", account.ID+100, originKey.UserID, 0)

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1beta/fileSearchStores/store-1:importFile", strings.NewReader(`{"fileName":"files/file-1"}`))
	c.Params = gin.Params{{Key: "storeAction", Value: "/store-1:importFile"}}
	attachGeminiAuthContext(c, originKey, originKey.UserID)

	h.GeminiV1BetaFileSearchStoreAction(c)
	require.Equal(t, http.StatusConflict, rec.Code)
	require.Contains(t, rec.Body.String(), "different upstream account")
}

func TestGeminiBatch_FiftyPercentDiscountAndIdempotentSettlement(t *testing.T) {
	// 1. Enqueue batch returning PENDING
	h, _, _, usageRepo, upstream, _, originKey, pollingKey := newInteractionBillingTestHandler(
		t,
		`{"name":"batches/b-discount-1","state":"BATCH_STATE_PENDING"}`,
	)

	recCreate := httptest.NewRecorder()
	cCreate, _ := gin.CreateTestContext(recCreate)
	cCreate.Request = httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-3.5-flash-lite:batchGenerateContent", strings.NewReader(`{"batch":{"displayName":"job1","inputConfig":{"requests":{"requests":[{"request":{"contents":[{"parts":[{"text":"hi"}]}]}}]}}}}`))
	cCreate.Params = gin.Params{{Key: "modelAction", Value: "/gemini-3.5-flash-lite:batchGenerateContent"}}
	attachGeminiAuthContext(cCreate, originKey, originKey.UserID)

	h.GeminiV1BetaModels(cCreate)
	require.Equal(t, http.StatusOK, recCreate.Code)
	require.Contains(t, recCreate.Body.String(), "batches/b-discount-1")
	require.Empty(t, usageRepo.snapshot(), "pending batch must not bill before completion")

	// 2. First poll returns SUCCEEDED with 1M input + 500k output tokens.
	// Base cost in test fixture is $1/1M input + $2/1M output = $1 + $1 = $2.00; 50% batch discount = $1.00.
	upstream.setBody(`{
		"name":"batches/b-discount-1",
		"state":"BATCH_STATE_SUCCEEDED",
		"output":{"inlinedResponses":{"inlinedResponses":[{"response":{"usageMetadata":{"promptTokenCount":1000000,"candidatesTokenCount":500000}}}]}}
	}`)

	recPoll1 := httptest.NewRecorder()
	cPoll1, _ := gin.CreateTestContext(recPoll1)
	cPoll1.Request = httptest.NewRequest(http.MethodGet, "/v1beta/batches/b-discount-1", nil)
	cPoll1.Params = gin.Params{{Key: "batchAction", Value: "/b-discount-1"}}
	attachGeminiAuthContext(cPoll1, pollingKey, pollingKey.UserID)

	h.GeminiV1BetaBatchAction(cPoll1)
	require.Equal(t, http.StatusOK, recPoll1.Code)

	logs := usageRepo.snapshot()
	require.Len(t, logs, 1)
	require.Equal(t, 1000000, logs[0].InputTokens)
	require.Equal(t, 500000, logs[0].OutputTokens)
	require.InDelta(t, 1.0, logs[0].TotalCost, 1e-6)
	require.InDelta(t, 1.0, logs[0].ActualCost, 1e-6)
	require.Equal(t, originKey.ID, logs[0].APIKeyID)

	// 3. Second poll & sweeper tick must NOT double-bill.
	recPoll2 := httptest.NewRecorder()
	cPoll2, _ := gin.CreateTestContext(recPoll2)
	cPoll2.Request = httptest.NewRequest(http.MethodGet, "/v1beta/batches/b-discount-1", nil)
	cPoll2.Params = gin.Params{{Key: "batchAction", Value: "/b-discount-1"}}
	attachGeminiAuthContext(cPoll2, pollingKey, pollingKey.UserID)

	h.GeminiV1BetaBatchAction(cPoll2)
	h.reconcilePendingGeminiBatch(context.Background(), "b-discount-1")
	require.Len(t, usageRepo.snapshot(), 1)
}

func TestGeminiBatch_SuccessWithoutUsageRemainsUnsettled(t *testing.T) {
	h, _, _, usageRepo, upstream, _, originKey, _ := newInteractionBillingTestHandler(
		t, `{"name":"batches/b-missing-usage","state":"BATCH_STATE_PENDING"}`,
	)
	createRec := httptest.NewRecorder()
	createCtx, _ := gin.CreateTestContext(createRec)
	createCtx.Request = httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-3.5-flash-lite:batchGenerateContent",
		strings.NewReader(`{"batch":{"inputConfig":{"requests":{"requests":[{"request":{"contents":[{"parts":[{"text":"hi"}]}]}}]}}}}`))
	createCtx.Params = gin.Params{{Key: "modelAction", Value: "/gemini-3.5-flash-lite:batchGenerateContent"}}
	attachGeminiAuthContext(createCtx, originKey, originKey.UserID)
	h.GeminiV1BetaModels(createCtx)
	require.Equal(t, http.StatusOK, createRec.Code)

	upstream.setBody(`{"name":"batches/b-missing-usage","state":"BATCH_STATE_SUCCEEDED"}`)
	poll := func() int {
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodGet, "/v1beta/batches/b-missing-usage", nil)
		c.Params = gin.Params{{Key: "batchAction", Value: "/b-missing-usage"}}
		attachGeminiAuthContext(c, originKey, originKey.UserID)
		h.GeminiV1BetaBatchAction(c)
		return rec.Code
	}
	require.Equal(t, http.StatusOK, poll())
	require.Empty(t, usageRepo.snapshot())
	settled, err := h.geminiCompatService.IsGeminiBatchSettled(context.Background(), "b-missing-usage")
	require.NoError(t, err)
	require.False(t, settled)

	upstream.setBody(`{"name":"batches/b-missing-usage","state":"BATCH_STATE_SUCCEEDED","output":{"inlinedResponses":{"inlinedResponses":[{"response":{"usageMetadata":{"promptTokenCount":1000000,"candidatesTokenCount":500000}}}]}}}`)
	require.Equal(t, http.StatusOK, poll())
	require.Len(t, usageRepo.snapshot(), 1)
}

func TestGeminiExtEndpoints_ListFilteringAndOpaqueAuthToken(t *testing.T) {
	h, _, _, _, upstream, account, originKey, _ := newInteractionBillingTestHandler(t, `{"agents":[{"name":"agents/agent-own"},{"name":"agents/agent-other"}]}`)
	h.geminiCompatService.BindGeminiExtResourceAccount(context.Background(), "agent", "agent-own", account.ID, originKey.UserID, 0)
	h.geminiCompatService.BindGeminiExtResourceAccount(context.Background(), "agent", "agent-other", account.ID, originKey.UserID+555, 0)

	recList := httptest.NewRecorder()
	cList, _ := gin.CreateTestContext(recList)
	cList.Request = httptest.NewRequest(http.MethodGet, "/v1beta/agents", nil)
	attachGeminiAuthContext(cList, originKey, originKey.UserID)

	h.GeminiV1BetaAgents(cList)
	require.Equal(t, http.StatusOK, recList.Code)
	require.Contains(t, recList.Body.String(), "agents/agent-own")
	require.NotContains(t, recList.Body.String(), "agents/agent-other")

	// Auth token creation returns opaque gw-* token instead of raw upstream token
	upstream.setBody(`{"name":"auth_tokens/upstream-secret-token-123","expireTime":"2030-01-01T00:00:00Z"}`)
	recTok := httptest.NewRecorder()
	cTok, _ := gin.CreateTestContext(recTok)
	cTok.Request = httptest.NewRequest(http.MethodPost, "/v1beta/auth_tokens", strings.NewReader(`{"uses":1}`))
	attachGeminiAuthContext(cTok, originKey, originKey.UserID)

	h.GeminiV1BetaAuthTokensCreate(cTok)
	require.Equal(t, http.StatusOK, recTok.Code)
	require.NotContains(t, recTok.Body.String(), "upstream-secret-token-123")
	require.Contains(t, recTok.Body.String(), "auth_tokens/gw-")
}

func TestGeminiBatch_PreDeleteSettlementAndStatsSingleDiscount(t *testing.T) {
	h, _, _, usageRepo, upstream, _, originKey, _ := newInteractionBillingTestHandler(
		t,
		`{"name":"batches/b-del-1","state":"BATCH_STATE_PENDING"}`,
	)

	recCreate := httptest.NewRecorder()
	cCreate, _ := gin.CreateTestContext(recCreate)
	cCreate.Request = httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-3.5-flash-lite:batchGenerateContent", strings.NewReader(`{"batch":{"inputConfig":{"requests":{"requests":[{"request":{"contents":[{"parts":[{"text":"hi"}]}]}}]}}}}`))
	cCreate.Params = gin.Params{{Key: "modelAction", Value: "/gemini-3.5-flash-lite:batchGenerateContent"}}
	attachGeminiAuthContext(cCreate, originKey, originKey.UserID)

	h.GeminiV1BetaModels(cCreate)
	require.Equal(t, http.StatusOK, recCreate.Code)

	// Batch completed before client calls DELETE; pre-delete snapshot settlement must capture the 50% discounted cost once.
	upstream.setBody(`{
		"name":"batches/b-del-1",
		"state":"BATCH_STATE_SUCCEEDED",
		"output":{"inlinedResponses":{"inlinedResponses":[{"response":{"usageMetadata":{"promptTokenCount":1000000,"candidatesTokenCount":500000}}}]}}
	}`)
	recDel := httptest.NewRecorder()
	cDel, _ := gin.CreateTestContext(recDel)
	cDel.Request = httptest.NewRequest(http.MethodDelete, "/v1beta/batches/b-del-1", nil)
	cDel.Params = gin.Params{{Key: "batchAction", Value: "/b-del-1"}}
	attachGeminiAuthContext(cDel, originKey, originKey.UserID)

	h.GeminiV1BetaBatchAction(cDel)
	require.Equal(t, http.StatusOK, recDel.Code)
	require.Equal(t, http.MethodDelete, upstream.lastReq.Method)

	logs := usageRepo.snapshot()
	require.Len(t, logs, 1)
	require.InDelta(t, 1.0, logs[0].TotalCost, 1e-6)
}
