//go:build unit

package handler

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type handlerInteractionCacheStub struct {
	mu       sync.Mutex
	bindings map[string]int64
	blobs    map[string][]byte
	claims   map[string]time.Time
	getErr   error
}

func (c *handlerInteractionCacheStub) GetSessionAccountID(_ context.Context, _ int64, sessionHash string) (int64, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.getErr != nil {
		return 0, c.getErr
	}
	if id, ok := c.bindings[sessionHash]; ok {
		return id, nil
	}
	return 0, service.ErrStickySessionNotFound
}

func (c *handlerInteractionCacheStub) SetSessionAccountID(_ context.Context, _ int64, sessionHash string, accountID int64, _ time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.bindings == nil {
		c.bindings = make(map[string]int64)
	}
	c.bindings[sessionHash] = accountID
	return nil
}

func (c *handlerInteractionCacheStub) RefreshSessionTTL(context.Context, int64, string, time.Duration) error {
	return nil
}

func (c *handlerInteractionCacheStub) DeleteSessionAccountID(context.Context, int64, string) error {
	return nil
}

func (c *handlerInteractionCacheStub) SetGrokVideoPendingBilling(_ context.Context, key string, body []byte, _ time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.blobs == nil {
		c.blobs = make(map[string][]byte)
	}
	cp := make([]byte, len(body))
	copy(cp, body)
	c.blobs[key] = cp
	return nil
}

func (c *handlerInteractionCacheStub) GetGrokVideoPendingBilling(_ context.Context, key string) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	val, ok := c.blobs[key]
	if !ok {
		return nil, nil
	}
	cp := make([]byte, len(val))
	copy(cp, val)
	return cp, nil
}

func (c *handlerInteractionCacheStub) ClaimGrokVideoBilled(_ context.Context, key string, ttl time.Duration) (bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.claims == nil {
		c.claims = make(map[string]time.Time)
	}
	now := time.Now()
	if exp, exists := c.claims[key]; exists && exp.After(now) {
		return false, nil
	}
	if ttl <= 0 {
		ttl = time.Minute
	}
	c.claims[key] = now.Add(ttl)
	return true, nil
}

func (c *handlerInteractionCacheStub) ReleaseGrokVideoBilled(_ context.Context, key string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.claims, key)
	return nil
}

func (c *handlerInteractionCacheStub) SetReasoningContent(context.Context, string, string, time.Duration) error {
	return nil
}

func (c *handlerInteractionCacheStub) GetReasoningContent(context.Context, string) (string, error) {
	return "", nil
}

func TestGeminiInteractionsDecisionsHandler_InputValidationAndIsolation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	newCtx := func(method, path, body string, groupPlatform string) (*gin.Context, *httptest.ResponseRecorder) {
		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(method, path, strings.NewReader(body))
		gid := int64(12)
		c.Set(string(middleware.ContextKeyAPIKey), &service.APIKey{
			ID:      1,
			UserID:  10,
			GroupID: &gid,
			Group:   &service.Group{ID: gid, Platform: groupPlatform},
		})
		c.Set(string(middleware.ContextKeyUser), middleware.AuthSubject{UserID: 10, Concurrency: 1})
		return c, rec
	}

	t.Run("decisions rejects missing model", func(t *testing.T) {
		c, rec := newCtx(http.MethodPost, "/v1beta/decisions", `{"input":"hi"}`, service.PlatformGemini)
		(&GatewayHandler{}).GeminiV1BetaDecisions(c)
		require.Equal(t, http.StatusBadRequest, rec.Code)
		require.Contains(t, rec.Body.String(), "model is required")
	})

	t.Run("interactions rejects missing model and agent", func(t *testing.T) {
		c, rec := newCtx(http.MethodPost, "/v1beta/interactions", `{"input":"hi"}`, service.PlatformGemini)
		(&GatewayHandler{}).GeminiV1BetaInteractionsCreate(c)
		require.Equal(t, http.StatusBadRequest, rec.Code)
		require.Contains(t, rec.Body.String(), "model or agent is required")
	})

	t.Run("interactions rejects composite antigravity platform", func(t *testing.T) {
		c, rec := newCtx(http.MethodPost, "/v1beta/interactions", `{"model":"gemini-3.5-flash-lite","input":"hi"}`, service.PlatformComposite)
		c.Request = c.Request.WithContext(service.WithCompositeRouteDecision(c.Request.Context(), service.CompositeRouteDecision{
			Matched:        true,
			TargetPlatform: service.PlatformAntigravity,
			UpstreamModel:  "gemini-3.8-flash",
		}))
		(&GatewayHandler{}).GeminiV1BetaInteractionsCreate(c)
		require.Equal(t, http.StatusBadRequest, rec.Code)
		require.Contains(t, rec.Body.String(), "Gemini AI Studio")
	})

	t.Run("interactions get rejects unsafe id", func(t *testing.T) {
		c, rec := newCtx(http.MethodGet, "/v1beta/interactions/..", "", service.PlatformGemini)
		c.Params = gin.Params{{Key: "interactionID", Value: ".."}}
		(&GatewayHandler{}).GeminiV1BetaInteractionsGet(c)
		require.Equal(t, http.StatusBadRequest, rec.Code)
		require.Contains(t, rec.Body.String(), "Invalid interaction ID")
	})

	t.Run("interactions get denies foreign owner", func(t *testing.T) {
		cache := &handlerInteractionCacheStub{}
		compat := service.NewGeminiMessagesCompatService(nil, nil, cache, nil, nil, nil, nil, nil, nil)
		gid := int64(12)
		require.NoError(t, compat.BindGeminiInteractionAccount(context.Background(), &gid, 999, "v1_foreign_1", 55))

		c, rec := newCtx(http.MethodGet, "/v1beta/interactions/v1_foreign_1", "", service.PlatformGemini)
		c.Params = gin.Params{{Key: "interactionID", Value: "v1_foreign_1"}}
		(&GatewayHandler{geminiCompatService: compat}).GeminiV1BetaInteractionsGet(c)
		require.Equal(t, http.StatusNotFound, rec.Code)
		require.Contains(t, rec.Body.String(), "Interaction not found")
	})

	t.Run("interactions get fails closed when binding is missing or expired", func(t *testing.T) {
		cache := &handlerInteractionCacheStub{}
		compat := service.NewGeminiMessagesCompatService(nil, nil, cache, nil, nil, nil, nil, nil, nil)

		c, rec := newCtx(http.MethodGet, "/v1beta/interactions/v1_expired_1", "", service.PlatformGemini)
		c.Params = gin.Params{{Key: "interactionID", Value: "v1_expired_1"}}
		(&GatewayHandler{geminiCompatService: compat}).GeminiV1BetaInteractionsGet(c)
		require.Equal(t, http.StatusNotFound, rec.Code)
		require.Contains(t, rec.Body.String(), "Interaction not found")
	})

	t.Run("interactions get returns 503 when binding cache lookup errors", func(t *testing.T) {
		cache := &handlerInteractionCacheStub{getErr: context.DeadlineExceeded}
		compat := service.NewGeminiMessagesCompatService(nil, nil, cache, nil, nil, nil, nil, nil, nil)

		c, rec := newCtx(http.MethodGet, "/v1beta/interactions/v1_err_1", "", service.PlatformGemini)
		c.Params = gin.Params{{Key: "interactionID", Value: "v1_err_1"}}
		(&GatewayHandler{geminiCompatService: compat}).GeminiV1BetaInteractionsGet(c)
		require.Equal(t, http.StatusServiceUnavailable, rec.Code)
		require.Contains(t, rec.Body.String(), "temporarily unavailable")
	})

	t.Run("interactions create rejects image generation when group disallows it", func(t *testing.T) {
		c, rec := newCtx(http.MethodPost, "/v1beta/interactions", `{"model":"gemini-3-pro-image-preview","input":"draw a cat"}`, service.PlatformGemini)
		(&GatewayHandler{gatewayService: &service.GatewayService{}}).GeminiV1BetaInteractionsCreate(c)
		require.Equal(t, http.StatusBadRequest, rec.Code)
		require.Contains(t, rec.Body.String(), service.ImageGenerationPermissionMessage())
	})

	t.Run("billable result recognizes image outputs even with zero token counts", func(t *testing.T) {
		require.True(t, hasBillableGeminiResult(&service.ForwardResult{ImageCount: 1}))
		require.False(t, hasBillableGeminiResult(&service.ForwardResult{}))
	})
}

type interactionTestAPIKeyRepo struct {
	service.APIKeyRepository
	keys map[int64]*service.APIKey
}

func (r *interactionTestAPIKeyRepo) GetByID(_ context.Context, id int64) (*service.APIKey, error) {
	if k, ok := r.keys[id]; ok {
		cp := *k
		return &cp, nil
	}
	return nil, service.ErrAPIKeyNotFound
}

type interactionTestUsageBillingRepo struct {
	service.UsageBillingRepository
	mu           sync.Mutex
	applyErrLeft int
	applyCalls   int
}

func (r *interactionTestUsageBillingRepo) Apply(_ context.Context, _ *service.UsageBillingCommand) (*service.UsageBillingApplyResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.applyCalls++
	if r.applyErrLeft > 0 {
		r.applyErrLeft--
		return nil, errors.New("transient billing tx failure")
	}
	return &service.UsageBillingApplyResult{Applied: true}, nil
}

type interactionTestUserRepo struct {
	service.UserRepository
	mu      sync.Mutex
	balance float64
}

func (r *interactionTestUserRepo) GetByID(_ context.Context, id int64) (*service.User, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return &service.User{ID: id, Balance: r.balance}, nil
}

func (r *interactionTestUserRepo) DeductBalance(_ context.Context, _ int64, amount float64) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.balance -= amount
	return nil
}

type interactionTestUsageRepo struct {
	service.UsageLogRepository
	mu   sync.Mutex
	logs []*service.UsageLog
}

func (r *interactionTestUsageRepo) Create(_ context.Context, log *service.UsageLog) (bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.logs = append(r.logs, log)
	return true, nil
}

func (r *interactionTestUsageRepo) snapshot() []*service.UsageLog {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]*service.UsageLog, len(r.logs))
	copy(out, r.logs)
	return out
}

type interactionTestUpstream struct {
	mu      sync.Mutex
	status  int
	body    string
	lastReq *http.Request
}

func (u *interactionTestUpstream) setBody(body string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.status = http.StatusOK
	u.body = body
}

func (u *interactionTestUpstream) setResponse(status int, body string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.status = status
	u.body = body
}

func (u *interactionTestUpstream) Do(req *http.Request, _ string, _ int64, _ int) (*http.Response, error) {
	u.mu.Lock()
	u.lastReq = req
	status := u.status
	if status == 0 {
		status = http.StatusOK
	}
	body := u.body
	u.mu.Unlock()
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"req-interaction-1"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}, nil
}

func (u *interactionTestUpstream) DoWithTLS(req *http.Request, proxyURL string, accountID int64, concurrency int, _ *tlsfingerprint.Profile) (*http.Response, error) {
	return u.Do(req, proxyURL, accountID, concurrency)
}

func newInteractionBillingTestHandler(t *testing.T, body string) (*GatewayHandler, *handlerInteractionCacheStub, *interactionTestUsageBillingRepo, *interactionTestUsageRepo, *interactionTestUpstream, *service.Account, *service.APIKey, *service.APIKey) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(dir, "model_pricing.json"), []byte(`{
		"gemini-3.5-flash-lite":{"litellm_provider":"gemini","mode":"chat","input_cost_per_token":0.000001,"output_cost_per_token":0.000002},
		"client-retry-alias":{"litellm_provider":"gemini","mode":"chat","input_cost_per_token":0.000001,"output_cost_per_token":0.000002},
		"gemini-4-hydrogen":{"litellm_provider":"gemini","mode":"chat","input_cost_per_token":0.000001,"output_cost_per_token":0.000002}
	}`), 0o644))
	cfg := &config.Config{Default: config.DefaultConfig{RateMultiplier: 1}}
	cfg.Pricing.DataDir = dir
	cfg.Pricing.UpdateIntervalHours = 100000
	pricingSvc := service.NewPricingService(cfg, nil)
	require.NoError(t, pricingSvc.Initialize())

	gid := int64(42)
	user := &service.User{ID: 10, Balance: 100}
	group := &service.Group{ID: gid, Hydrated: true, Platform: service.PlatformGemini, Status: service.StatusActive, RateMultiplier: 1}
	originKey := &service.APIKey{ID: 101, UserID: 10, User: user, GroupID: &gid, Group: group}
	pollingKey := &service.APIKey{ID: 202, UserID: 10, User: user, GroupID: &gid, Group: group}
	account := &service.Account{
		ID:          7,
		Platform:    service.PlatformGemini,
		Type:        service.AccountTypeAPIKey,
		Credentials: map[string]any{"api_key": "sk-goog"},
		Status:      service.StatusActive,
		Schedulable: true,
		Concurrency: 5,
		Priority:    1,
		AccountGroups: []service.AccountGroup{
			{AccountID: 7, GroupID: gid},
		},
	}

	cache := &handlerInteractionCacheStub{}
	userRepo := &interactionTestUserRepo{balance: 100}
	usageBillingRepo := &interactionTestUsageBillingRepo{}
	usageRepo := &interactionTestUsageRepo{}
	keyRepo := &interactionTestAPIKeyRepo{keys: map[int64]*service.APIKey{101: originKey, 202: pollingKey}}
	apiKeySvc := service.NewAPIKeyService(keyRepo, nil, nil, nil, nil, nil, cfg)
	bcs := service.NewBillingCacheService(nil, userRepo, nil, nil, nil, nil, cfg, nil)
	t.Cleanup(bcs.Stop)

	snapshot := service.NewSchedulerSnapshotService(&fakeSchedulerCache{accounts: []*service.Account{account}}, nil, nil, nil, nil)
	gw := service.NewGatewayService(
		nil,
		&fakeGroupRepo{group: group},
		usageRepo,
		usageBillingRepo,
		userRepo,
		nil, nil, nil,
		cfg,
		snapshot,
		nil,
		service.NewBillingService(cfg, pricingSvc),
		nil,
		bcs,
		nil, nil,
		&service.DeferredService{},
		nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil,
	)
	upstream := &interactionTestUpstream{body: body}
	compat := service.NewGeminiMessagesCompatService(&cachedBillingAccountRepo{accounts: map[int64]*service.Account{7: account}}, nil, cache, nil, nil, nil, upstream, nil, cfg)
	h := &GatewayHandler{
		gatewayService:      gw,
		geminiCompatService: compat,
		billingCacheService: bcs,
		apiKeyService:       apiKeySvc,
		concurrencyHelper:   NewConcurrencyHelper(service.NewConcurrencyService(&fakeConcurrencyCache{}), SSEPingFormatNone, 0),
		cfg:                 cfg,
	}
	t.Cleanup(h.StopGeminiInteractionPendingSweeper)
	return h, cache, usageBillingRepo, usageRepo, upstream, account, originKey, pollingKey
}

func TestGeminiInteractionsDeferredBilling_SettlementFixes(t *testing.T) {
	t.Run("GET settlement attributes originating APIKey and unconditionally restores creation models", func(t *testing.T) {
		h, _, _, usageRepo, _, account, originKey, pollingKey := newInteractionBillingTestHandler(
			t,
			`{"id":"v1_origin_bill","status":"completed","model":"gemini-3.5-flash-lite-001","usage":{"total_input_tokens":120,"total_output_tokens":30}}`,
		)
		gid := *originKey.GroupID
		require.NoError(t, h.geminiCompatService.BindGeminiInteractionAccount(context.Background(), &gid, originKey.UserID, "v1_origin_bill", account.ID))
		require.NoError(t, h.geminiCompatService.SaveGeminiInteractionMeta(context.Background(), "v1_origin_bill", service.GeminiInteractionMeta{
			RequestModel:  "custom-gemini-alias",
			UpstreamModel: "gemini-3.5-flash-lite",
			APIKeyID:      originKey.ID,
			UserID:        originKey.UserID,
			AccountID:     account.ID,
			ChannelUsageFields: service.ChannelUsageFields{
				OriginalModel:      "custom-gemini-alias",
				ChannelMappedModel: "gemini-3.5-flash-lite",
			},
			PricingAtUnixNano: time.Now().UnixNano(),
		}))

		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodGet, "/v1beta/interactions/v1_origin_bill", nil)
		c.Params = gin.Params{{Key: "interactionID", Value: "v1_origin_bill"}}
		c.Set(string(middleware.ContextKeyAPIKey), pollingKey)
		c.Set(string(middleware.ContextKeyUser), middleware.AuthSubject{UserID: pollingKey.UserID, Concurrency: 1})

		h.GeminiV1BetaInteractionsGet(c)
		require.Equal(t, http.StatusOK, rec.Code)

		logs := usageRepo.snapshot()
		require.Len(t, logs, 1)
		require.Equal(t, originKey.ID, logs[0].APIKeyID)
		require.Equal(t, "custom-gemini-alias", logs[0].Model)
		require.NotNil(t, logs[0].UpstreamModel)
		require.Equal(t, "gemini-3.5-flash-lite", *logs[0].UpstreamModel)
		require.NotNil(t, logs[0].UpstreamResponseModel)
		require.Equal(t, "gemini-3.5-flash-lite-001", *logs[0].UpstreamResponseModel)
	})

	t.Run("reconcileGeminiInteractionUsage retries when RecordUsage fails transiently", func(t *testing.T) {
		origIntervals := geminiInteractionReconcileIntervals
		origMaxWait := geminiInteractionReconcileMaxWait
		geminiInteractionReconcileIntervals = []time.Duration{2 * time.Millisecond}
		geminiInteractionReconcileMaxWait = 500 * time.Millisecond
		defer func() {
			geminiInteractionReconcileIntervals = origIntervals
			geminiInteractionReconcileMaxWait = origMaxWait
		}()

		h, _, usageBillingRepo, usageRepo, _, account, originKey, _ := newInteractionBillingTestHandler(
			t,
			`{"id":"v1_retry_record","status":"completed","model":"upstream-actual-model","usage":{"total_input_tokens":80,"total_output_tokens":20}}`,
		)
		usageBillingRepo.applyErrLeft = 2

		snap := geminiInteractionUsageSnapshot{
			parentCtx:     context.Background(),
			apiKey:        originKey,
			account:       account,
			userID:        originKey.UserID,
			pricingAt:     time.Now(),
			requestModel:  "client-retry-alias",
			upstreamModel: "gemini-3.5-flash-lite",
		}
		h.reconcileGeminiInteractionUsage(snap, "v1_retry_record", nil)

		require.Equal(t, 3, usageBillingRepo.applyCalls)
		logs := usageRepo.snapshot()
		require.Len(t, logs, 3) // 2 failed-billing audit logs (ActualCost=0) + 1 settled log
		settledLog := logs[len(logs)-1]
		require.Greater(t, settledLog.ActualCost, 0.0)
		require.Equal(t, originKey.ID, settledLog.APIKeyID)
		require.Equal(t, "client-retry-alias", settledLog.Model)
		require.NotNil(t, settledLog.UpstreamModel)
		require.Equal(t, "gemini-3.5-flash-lite", *settledLog.UpstreamModel)
		require.NotNil(t, settledLog.UpstreamResponseModel)
		require.Equal(t, "upstream-actual-model", *settledLog.UpstreamResponseModel)

		settled, err := h.geminiCompatService.IsGeminiInteractionSettled(context.Background(), "v1_retry_record")
		require.NoError(t, err)
		require.True(t, settled)
	})

	t.Run("retains pending billing beyond fast reconcile deadline and settles via durable sweep", func(t *testing.T) {
		origIntervals := geminiInteractionReconcileIntervals
		origMaxWait := geminiInteractionReconcileMaxWait
		origSlow := geminiInteractionSlowPollReschedule
		geminiInteractionReconcileIntervals = []time.Duration{4 * time.Millisecond}
		geminiInteractionReconcileMaxWait = 12 * time.Millisecond
		geminiInteractionSlowPollReschedule = 5 * time.Millisecond
		defer func() {
			geminiInteractionReconcileIntervals = origIntervals
			geminiInteractionReconcileMaxWait = origMaxWait
			geminiInteractionSlowPollReschedule = origSlow
		}()

		h, _, _, usageRepo, upstream, account, originKey, _ := newInteractionBillingTestHandler(
			t,
			`{"id":"v1_slow_sweep","status":"in_progress"}`,
		)
		gid := *originKey.GroupID
		require.NoError(t, h.geminiCompatService.BindGeminiInteractionAccount(context.Background(), &gid, originKey.UserID, "v1_slow_sweep", account.ID))
		require.NoError(t, h.geminiCompatService.SaveGeminiInteractionMeta(context.Background(), "v1_slow_sweep", service.GeminiInteractionMeta{
			RequestModel:      "deep-research-alias",
			UpstreamModel:     "gemini-4-hydrogen",
			APIKeyID:          originKey.ID,
			UserID:            originKey.UserID,
			AccountID:         account.ID,
			PricingAtUnixNano: time.Now().UnixNano(),
		}))

		inflightReleased := false
		snap := geminiInteractionUsageSnapshot{
			parentCtx:     context.Background(),
			apiKey:        originKey,
			account:       account,
			userID:        originKey.UserID,
			pricingAt:     time.Now(),
			requestModel:  "deep-research-alias",
			upstreamModel: "gemini-4-hydrogen",
		}
		h.reconcileGeminiInteractionUsage(snap, "v1_slow_sweep", func() {
			inflightReleased = true
		})
		require.True(t, inflightReleased)
		require.Empty(t, usageRepo.snapshot())

		time.Sleep(15 * time.Millisecond)
		due, err := h.geminiCompatService.ListDueGeminiInteractionPending(context.Background(), time.Now(), 10)
		require.NoError(t, err)
		require.Contains(t, due, "v1_slow_sweep")

		upstream.setBody(`{"id":"v1_slow_sweep","status":"completed","agent":"deep-research-pro","usage":{"total_input_tokens":500,"total_output_tokens":150}}`)
		h.ReconcilePendingGeminiInteractions(context.Background())

		logs := usageRepo.snapshot()
		require.Len(t, logs, 1)
		require.Equal(t, originKey.ID, logs[0].APIKeyID)
		require.Equal(t, "deep-research-alias", logs[0].Model)
		require.NotNil(t, logs[0].UpstreamModel)
		require.Equal(t, "gemini-4-hydrogen", *logs[0].UpstreamModel)
		require.NotNil(t, logs[0].UpstreamResponseModel)
		require.Equal(t, "deep-research-pro", *logs[0].UpstreamResponseModel)

		dueAfter, err := h.geminiCompatService.ListDueGeminiInteractionPending(context.Background(), time.Now().Add(time.Minute), 10)
		require.NoError(t, err)
		require.NotContains(t, dueAfter, "v1_slow_sweep")
	})

	t.Run("DELETE settles completed unbilled usage before deleting upstream", func(t *testing.T) {
		h, _, _, usageRepo, _, account, originKey, _ := newInteractionBillingTestHandler(
			t,
			`{"id":"v1_del_settle","status":"completed","model":"gemini-3.5-flash-lite","usage":{"total_input_tokens":140,"total_output_tokens":45}}`,
		)
		gid := *originKey.GroupID
		require.NoError(t, h.geminiCompatService.BindGeminiInteractionAccount(context.Background(), &gid, originKey.UserID, "v1_del_settle", account.ID))
		require.NoError(t, h.geminiCompatService.SaveGeminiInteractionMeta(context.Background(), "v1_del_settle", service.GeminiInteractionMeta{
			RequestModel:      "alias-before-delete",
			UpstreamModel:     "gemini-3.5-flash-lite",
			APIKeyID:          originKey.ID,
			UserID:            originKey.UserID,
			AccountID:         account.ID,
			PricingAtUnixNano: time.Now().UnixNano(),
		}))
		require.NoError(t, h.geminiCompatService.ScheduleGeminiInteractionPending(context.Background(), "v1_del_settle", time.Now().Add(time.Minute)))

		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodDelete, "/v1beta/interactions/v1_del_settle", nil)
		c.Params = gin.Params{{Key: "interactionID", Value: "v1_del_settle"}}
		c.Set(string(middleware.ContextKeyAPIKey), originKey)
		c.Set(string(middleware.ContextKeyUser), middleware.AuthSubject{UserID: originKey.UserID, Concurrency: 1})

		h.GeminiV1BetaInteractionsDelete(c)
		require.Equal(t, http.StatusOK, rec.Code)

		logs := usageRepo.snapshot()
		require.Len(t, logs, 1)
		require.Equal(t, originKey.ID, logs[0].APIKeyID)
		require.Equal(t, "alias-before-delete", logs[0].Model)
		require.Equal(t, "gemini_interaction:v1_del_settle", logs[0].RequestID)

		settled, err := h.geminiCompatService.IsGeminiInteractionSettled(context.Background(), "v1_del_settle")
		require.NoError(t, err)
		require.True(t, settled)
	})

	t.Run("reconcile without model field in snapshot does not leak status into UpstreamResponseModel", func(t *testing.T) {
		origIntervals := geminiInteractionReconcileIntervals
		origMaxWait := geminiInteractionReconcileMaxWait
		geminiInteractionReconcileIntervals = []time.Duration{2 * time.Millisecond}
		geminiInteractionReconcileMaxWait = 300 * time.Millisecond
		defer func() {
			geminiInteractionReconcileIntervals = origIntervals
			geminiInteractionReconcileMaxWait = origMaxWait
		}()

		h, _, _, usageRepo, _, account, originKey, _ := newInteractionBillingTestHandler(
			t,
			`{"id":"v1_no_model_snap","status":"completed","usage":{"total_input_tokens":60,"total_output_tokens":15}}`,
		)
		snap := geminiInteractionUsageSnapshot{
			parentCtx:     context.Background(),
			apiKey:        originKey,
			account:       account,
			userID:        originKey.UserID,
			pricingAt:     time.Now(),
			requestModel:  "gemini-3.5-flash-lite",
			upstreamModel: "gemini-3.5-flash-lite",
		}
		h.reconcileGeminiInteractionUsage(snap, "v1_no_model_snap", nil)

		logs := usageRepo.snapshot()
		require.Len(t, logs, 1)
		require.Nil(t, logs[0].UpstreamResponseModel)
		require.Equal(t, "gemini-3.5-flash-lite", logs[0].Model)
	})

	t.Run("DELETE returns 503 and preserves pending entry when pre-delete snapshot fails transiently", func(t *testing.T) {
		h, _, _, _, upstream, account, originKey, _ := newInteractionBillingTestHandler(t, `{}`)
		upstream.setResponse(http.StatusBadGateway, `{"error":{"message":"upstream busy"}}`)

		gid := *originKey.GroupID
		require.NoError(t, h.geminiCompatService.BindGeminiInteractionAccount(context.Background(), &gid, originKey.UserID, "v1_del_503", account.ID))
		require.NoError(t, h.geminiCompatService.ScheduleGeminiInteractionPending(context.Background(), "v1_del_503", time.Now().Add(-time.Second)))

		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodDelete, "/v1beta/interactions/v1_del_503", nil)
		c.Params = gin.Params{{Key: "interactionID", Value: "v1_del_503"}}
		c.Set(string(middleware.ContextKeyAPIKey), originKey)
		c.Set(string(middleware.ContextKeyUser), middleware.AuthSubject{UserID: originKey.UserID, Concurrency: 1})

		h.GeminiV1BetaInteractionsDelete(c)
		require.Equal(t, http.StatusServiceUnavailable, rec.Code)

		due, err := h.geminiCompatService.ListDueGeminiInteractionPending(context.Background(), time.Now(), 10)
		require.NoError(t, err)
		require.Contains(t, due, "v1_del_503")
	})
}

func TestParseAcceptPrefersSSE(t *testing.T) {
	require.False(t, parseAcceptPrefersSSE(""))
	require.False(t, parseAcceptPrefersSSE("*/*"))
	require.False(t, parseAcceptPrefersSSE("application/json;q=1, text/event-stream;q=0"))
	require.False(t, parseAcceptPrefersSSE("text/event-stream;q=0.0, application/json"))
	require.False(t, parseAcceptPrefersSSE("application/json, text/event-stream"))
	require.True(t, parseAcceptPrefersSSE("text/event-stream"))
	require.True(t, parseAcceptPrefersSSE("text/event-stream, application/json;q=0.8"))
}

func TestGeminiInteractionsGetStreamNegotiation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("GET with Accept q=0 for SSE uses non-streaming JSON", func(t *testing.T) {
		h, _, cache, usageRepo, upstream, account, originKey, _ := newInteractionBillingTestHandler(
			t,
			`{"id":"v1_get_json","model":"gemini-3.5-flash-lite","status":"completed","usage":{"total_input_tokens":44,"total_output_tokens":11}}`,
		)
		gid := *originKey.GroupID
		require.NoError(t, h.geminiCompatService.BindGeminiInteractionAccount(context.Background(), &gid, originKey.UserID, "v1_get_json", account.ID))
		require.NoError(t, h.geminiCompatService.SaveGeminiInteractionMeta(context.Background(), "v1_get_json", service.GeminiInteractionMeta{
			RequestModel:      "gemini-3.5-flash-lite",
			UpstreamModel:     "gemini-3.5-flash-lite",
			APIKeyID:          originKey.ID,
			UserID:            originKey.UserID,
			AccountID:         account.ID,
			PricingAtUnixNano: time.Now().UnixNano(),
		}))
		_ = cache

		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodGet, "/v1beta/interactions/v1_get_json", nil)
		c.Request.Header.Set("Accept", "application/json;q=1, text/event-stream;q=0")
		c.Params = gin.Params{{Key: "interactionID", Value: "v1_get_json"}}
		c.Set(string(middleware.ContextKeyAPIKey), originKey)
		c.Set(string(middleware.ContextKeyUser), middleware.AuthSubject{UserID: originKey.UserID, Concurrency: 1})

		h.GeminiV1BetaInteractionsGet(c)
		require.Equal(t, http.StatusOK, rec.Code)

		upstream.mu.Lock()
		lastAccept := upstream.lastReq.Header.Get("Accept")
		lastStreamQuery := upstream.lastReq.URL.Query().Get("stream")
		upstream.mu.Unlock()
		require.Empty(t, lastAccept)
		require.Empty(t, lastStreamQuery)

		logs := usageRepo.snapshot()
		require.Len(t, logs, 1)
		require.Equal(t, 44, logs[0].InputTokens)
		require.Equal(t, 11, logs[0].OutputTokens)
	})

	t.Run("GET with explicit stream=false overrides Accept text/event-stream", func(t *testing.T) {
		h, _, _, _, upstream, account, originKey, _ := newInteractionBillingTestHandler(
			t,
			`{"id":"v1_get_qfalse","model":"gemini-3.5-flash-lite","status":"completed"}`,
		)
		gid := *originKey.GroupID
		require.NoError(t, h.geminiCompatService.BindGeminiInteractionAccount(context.Background(), &gid, originKey.UserID, "v1_get_qfalse", account.ID))

		rec := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(rec)
		c.Request = httptest.NewRequest(http.MethodGet, "/v1beta/interactions/v1_get_qfalse?stream=false", nil)
		c.Request.Header.Set("Accept", "text/event-stream")
		c.Params = gin.Params{{Key: "interactionID", Value: "v1_get_qfalse"}}
		c.Set(string(middleware.ContextKeyAPIKey), originKey)
		c.Set(string(middleware.ContextKeyUser), middleware.AuthSubject{UserID: originKey.UserID, Concurrency: 1})

		h.GeminiV1BetaInteractionsGet(c)
		require.Equal(t, http.StatusOK, rec.Code)

		upstream.mu.Lock()
		lastAccept := upstream.lastReq.Header.Get("Accept")
		upstream.mu.Unlock()
		require.Empty(t, lastAccept)
	})
}

func TestGeminiInteractionsDelete_ActiveClaimLeaseReturns503(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h, _, _, _, _, account, originKey, _ := newInteractionBillingTestHandler(
		t,
		`{"id":"v1_del_lease","model":"gemini-3.5-flash-lite","status":"completed","usage":{"total_input_tokens":50,"total_output_tokens":10}}`,
	)
	gid := *originKey.GroupID
	require.NoError(t, h.geminiCompatService.BindGeminiInteractionAccount(context.Background(), &gid, originKey.UserID, "v1_del_lease", account.ID))
	require.NoError(t, h.geminiCompatService.ScheduleGeminiInteractionPending(context.Background(), "v1_del_lease", time.Now().Add(-time.Second)))

	// Another worker holds the active claim lease while RecordUsage has not finished (settled=false).
	claimed, err := h.geminiCompatService.ClaimGeminiInteractionBilled(context.Background(), "v1_del_lease")
	require.NoError(t, err)
	require.True(t, claimed)

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodDelete, "/v1beta/interactions/v1_del_lease", nil)
	c.Params = gin.Params{{Key: "interactionID", Value: "v1_del_lease"}}
	c.Set(string(middleware.ContextKeyAPIKey), originKey)
	c.Set(string(middleware.ContextKeyUser), middleware.AuthSubject{UserID: originKey.UserID, Concurrency: 1})

	h.GeminiV1BetaInteractionsDelete(c)
	require.Equal(t, http.StatusServiceUnavailable, rec.Code)

	due, err := h.geminiCompatService.ListDueGeminiInteractionPending(context.Background(), time.Now(), 10)
	require.NoError(t, err)
	require.Contains(t, due, "v1_del_lease")
}

func TestGeminiInteractionsPost_ExplicitStreamFalseAndQZeroAccept(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h, _, _, usageRepo, upstream, account, originKey, _ := newInteractionBillingTestHandler(
		t,
		`{"id":"v1_post_json","model":"gemini-3.5-flash-lite","status":"completed","usage":{"total_input_tokens":33,"total_output_tokens":7}}`,
	)

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	body := `{"input":"Reply with OK only","model":"gemini-3.5-flash-lite","stream":false}`
	c.Request = httptest.NewRequest(http.MethodPost, "/v1beta/interactions", strings.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")
	c.Request.Header.Set("Accept", "application/json;q=1, text/event-stream;q=0")
	c.Set(string(middleware.ContextKeyAPIKey), originKey)
	c.Set(string(middleware.ContextKeyUser), middleware.AuthSubject{UserID: originKey.UserID, Concurrency: 1})

	h.GeminiV1BetaInteractionsCreate(c)
	require.Equal(t, http.StatusOK, rec.Code)

	upstream.mu.Lock()
	lastAccept := upstream.lastReq.Header.Get("Accept")
	upstream.mu.Unlock()
	require.Empty(t, lastAccept)

	// Interaction owner/account/meta/settled bindings must be persisted even on non-streaming POST.
	acctID, verified, err := h.geminiCompatService.ResolveGeminiInteractionAccount(context.Background(), originKey.GroupID, originKey.UserID, "v1_post_json")
	require.NoError(t, err)
	require.True(t, verified)
	require.Equal(t, account.ID, acctID)

	settled, err := h.geminiCompatService.IsGeminiInteractionSettled(context.Background(), "v1_post_json")
	require.NoError(t, err)
	require.True(t, settled)

	logs := usageRepo.snapshot()
	require.Len(t, logs, 1)
	require.Equal(t, 33, logs[0].InputTokens)
	require.Equal(t, 7, logs[0].OutputTokens)
}
