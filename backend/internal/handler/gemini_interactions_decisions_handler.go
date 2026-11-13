package handler

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"

	pkghttputil "github.com/Wei-Shaw/sub2api/internal/pkg/httputil"
	"github.com/Wei-Shaw/sub2api/internal/pkg/ip"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"go.uber.org/zap"
)

const (
	geminiInteractionBindTimeout             = 5 * time.Second
	geminiInteractionPollTimeout             = 15 * time.Second
	defaultGeminiInteractionReconcileMaxWait = 20 * time.Minute
	geminiInteractionSlowReconcileInterval   = 2 * time.Minute
	geminiInteractionPendingSweepInterval    = 1 * time.Minute
	geminiInteractionPendingSweepBatchSize   = int64(50)
	geminiInteractionPendingMaxAge           = 7 * 24 * time.Hour
)

var (
	geminiInteractionReconcileIntervals = []time.Duration{
		2 * time.Second,
		4 * time.Second,
		8 * time.Second,
		15 * time.Second,
		30 * time.Second,
	}
	geminiInteractionReconcileMaxWait     = defaultGeminiInteractionReconcileMaxWait
	geminiInteractionPendingSweepEvery    = geminiInteractionPendingSweepInterval
	geminiInteractionSlowPollReschedule   = geminiInteractionSlowReconcileInterval
	geminiInteractionMandatoryWaitTimeout = 20 * time.Second
)

// GeminiV1BetaDecisions proxies POST /v1beta/decisions.
func (h *GatewayHandler) GeminiV1BetaDecisions(c *gin.Context) {
	h.handleGeminiAIStudioPost(c, true)
}

// GeminiV1BetaInteractionsCreate proxies POST /v1beta/interactions.
func (h *GatewayHandler) GeminiV1BetaInteractionsCreate(c *gin.Context) {
	h.handleGeminiAIStudioPost(c, false)
}

// GeminiV1BetaInteractionsGet proxies GET /v1beta/interactions/:interactionID.
func (h *GatewayHandler) GeminiV1BetaInteractionsGet(c *gin.Context) {
	stream := false
	if streamParam := strings.TrimSpace(c.Query("stream")); streamParam != "" {
		if parsed, err := strconv.ParseBool(streamParam); err == nil {
			stream = parsed
		} else {
			stream = strings.EqualFold(streamParam, "true")
		}
	} else {
		stream = parseAcceptPrefersSSE(c.GetHeader("Accept"))
	}
	h.handleGeminiInteractionByID(c, http.MethodGet, false, stream)
}

// parseAcceptPrefersSSE evaluates RFC-style Accept media ranges and q-weights,
// returning true only when text/event-stream has q > 0 and strictly beats application/json.
func parseAcceptPrefersSSE(acceptHeader string) bool {
	if acceptHeader == "" {
		return false
	}
	var (
		hasSSE  bool
		sseQ    float64
		hasJSON bool
		jsonQ   float64
	)
	for _, part := range strings.Split(acceptHeader, ",") {
		segments := strings.Split(strings.TrimSpace(part), ";")
		mediaType := strings.ToLower(strings.TrimSpace(segments[0]))
		if mediaType != "text/event-stream" && mediaType != "application/json" {
			continue
		}
		q := 1.0
		validQ := true
		for _, param := range segments[1:] {
			kv := strings.SplitN(strings.TrimSpace(param), "=", 2)
			if len(kv) != 2 || !strings.EqualFold(strings.TrimSpace(kv[0]), "q") {
				continue
			}
			parsed, err := strconv.ParseFloat(strings.TrimSpace(kv[1]), 64)
			if err != nil || parsed < 0 || parsed > 1 {
				validQ = false
				break
			}
			q = parsed
		}
		if !validQ {
			continue
		}
		if mediaType == "text/event-stream" && (!hasSSE || q > sseQ) {
			hasSSE = true
			sseQ = q
		} else if mediaType == "application/json" && (!hasJSON || q > jsonQ) {
			hasJSON = true
			jsonQ = q
		}
	}
	if !hasSSE || sseQ <= 0 {
		return false
	}
	if hasJSON && jsonQ >= sseQ {
		return false
	}
	return true
}

// GeminiV1BetaInteractionsCancel proxies POST /v1beta/interactions/:interactionID/cancel.
func (h *GatewayHandler) GeminiV1BetaInteractionsCancel(c *gin.Context) {
	h.handleGeminiInteractionByID(c, http.MethodPost, true, false)
}

// GeminiV1BetaInteractionsDelete proxies DELETE /v1beta/interactions/:interactionID.
func (h *GatewayHandler) GeminiV1BetaInteractionsDelete(c *gin.Context) {
	h.handleGeminiInteractionByID(c, http.MethodDelete, false, false)
}

func (h *GatewayHandler) handleGeminiAIStudioPost(c *gin.Context, isDecisions bool) {
	apiKey, ok := middleware.GetAPIKeyFromContext(c)
	if !ok || apiKey == nil {
		googleError(c, http.StatusUnauthorized, "Invalid API key")
		return
	}
	authSubject, ok := middleware.GetAuthSubjectFromContext(c)
	if !ok {
		googleError(c, http.StatusInternalServerError, "User context not found")
		return
	}

	component := "handler.gemini_v1beta.interactions"
	if isDecisions {
		component = "handler.gemini_v1beta.decisions"
	}
	reqLog := requestLogger(
		c,
		component,
		zap.Int64("user_id", authSubject.UserID),
		zap.Int64("api_key_id", apiKey.ID),
		zap.Any("group_id", apiKey.GroupID),
	)

	effectivePlatform := effectiveAPIKeyPlatform(c, apiKey)
	if !middleware.HasForcePlatform(c) && effectivePlatform != service.PlatformGemini {
		if servesCompositeAntigravityGemini(apiKey, effectivePlatform) {
			googleError(c, http.StatusBadRequest, "This endpoint requires a Gemini AI Studio account")
			return
		}
		googleError(c, http.StatusBadRequest, "API key group platform is not gemini")
		return
	}
	if middleware.HasForcePlatform(c) {
		forcePlatform, _ := middleware.GetForcePlatformFromContext(c)
		if forcePlatform != service.PlatformGemini {
			googleError(c, http.StatusBadRequest, "This endpoint requires a Gemini AI Studio account")
			return
		}
	}

	body, err := pkghttputil.ReadRequestBodyWithPrealloc(c.Request)
	if err != nil {
		if handleRequestBodyReadError(c, reqLog, err) {
			return
		}
		status, message := requestBodyReadFailure(reqLog, c.Request, err)
		googleErrorWithType(c, status, "invalid_request_error", "", message)
		return
	}
	if len(body) == 0 {
		googleError(c, http.StatusBadRequest, "Request body is empty")
		return
	}

	var (
		modelName             string
		previousInteractionID string
		stream                bool
	)
	if isDecisions {
		if !gjson.ValidBytes(body) {
			googleError(c, http.StatusBadRequest, "Invalid JSON request body")
			return
		}
		modelName = strings.TrimSpace(gjson.GetBytes(body, "model").String())
		if modelName == "" {
			googleError(c, http.StatusBadRequest, "model is required")
			return
		}
	} else {
		var validJSON, hasStreamField bool
		modelName, previousInteractionID, stream, hasStreamField, validJSON = service.ParseGeminiInteractionBodyWithStreamFlag(body)
		if !validJSON {
			googleError(c, http.StatusBadRequest, "Invalid JSON request body")
			return
		}
		if modelName == "" {
			googleError(c, http.StatusBadRequest, "model or agent is required")
			return
		}
		if !hasStreamField && parseAcceptPrefersSSE(c.GetHeader("Accept")) {
			stream = true
		}
		if previousInteractionID != "" && !service.IsSafeGeminiInteractionID(previousInteractionID) {
			googleError(c, http.StatusBadRequest, "Invalid previous_interaction_id")
			return
		}
	}

	clientRequestedModel := modelName
	if resolvedModel, ok := service.ResolvedUpstreamModelFromContext(c.Request.Context()); ok && strings.TrimSpace(resolvedModel) != "" {
		modelName = strings.TrimSpace(resolvedModel)
	}
	reqLog = reqLog.With(zap.String("model", modelName), zap.Bool("stream", stream))

	setOpsRequestContext(c, modelName, stream)
	setOpsEndpointContext(c, "", int16(service.RequestTypeFromLegacy(stream, false)))
	pricingCtx, pricingAt := service.WithGatewayTokenRequestPricing(c.Request.Context())
	c.Request = c.Request.WithContext(pricingCtx)

	channelMapping, _ := h.gatewayService.ResolveChannelMappingAndRestrict(c.Request.Context(), apiKey.GroupID, modelName)
	reqModel := modelName
	if !isDecisions && geminiImageGenerationRejected(c, apiKey, body, clientRequestedModel, reqModel, channelMapping.MappedModel) {
		googleError(c, http.StatusBadRequest, service.ImageGenerationPermissionMessage())
		return
	}
	auditProtocol := service.ContentModerationProtocolGeminiInteractions
	if isDecisions {
		auditProtocol = service.ContentModerationProtocolGeminiDecisions
	}
	if decision := h.checkSecurityAudit(c, reqLog, apiKey, authSubject, auditProtocol, reqModel, body); decision != nil && !decision.AllowNextStage {
		googleSecurityAuditError(c, decision)
		return
	}
	if channelMapping.Mapped {
		modelName = channelMapping.MappedModel
	}

	var boundAccountID int64
	if !isDecisions && previousInteractionID != "" {
		boundID, verified, resolveErr := h.geminiCompatService.ResolveGeminiInteractionAccount(
			c.Request.Context(),
			apiKey.GroupID,
			authSubject.UserID,
			previousInteractionID,
		)
		if resolveErr != nil {
			reqLog.Warn("gemini.interaction.resolve_previous_failed", zap.Error(resolveErr))
			googleError(c, http.StatusServiceUnavailable, "Interaction state temporarily unavailable")
			return
		}
		if !verified || boundID <= 0 {
			googleError(c, http.StatusNotFound, "Previous interaction not found")
			return
		}
		boundAccountID = boundID
	}

	subscription, _ := middleware.GetSubscriptionFromContext(c)
	geminiConcurrency := NewConcurrencyHelper(h.concurrencyHelper.concurrencyService, SSEPingFormatNone, 0)
	streamStarted := false
	if h.errorPassthroughService != nil {
		service.BindErrorPassthroughService(c, h.errorPassthroughService)
	}
	userReleaseFunc, err := geminiConcurrency.AcquireUserSlotWithWait(c, authSubject.UserID, authSubject.Concurrency, stream, &streamStarted)
	if err != nil {
		reqLog.Warn("gemini.aistudio_endpoint.user_slot_acquire_failed", zap.Error(err))
		googleConcurrencyError(c, err, "user")
		return
	}
	userReleaseFunc = wrapReleaseOnDone(c.Request.Context(), userReleaseFunc)
	if userReleaseFunc != nil {
		defer userReleaseFunc()
	}

	if err := h.billingCacheService.CheckBillingEligibility(c.Request.Context(), apiKey.User, apiKey, apiKey.Group, subscription, service.QuotaPlatform(c.Request.Context(), apiKey)); err != nil {
		reqLog.Info("gemini.aistudio_endpoint.billing_check_failed", zap.Error(err))
		status, code, message, retryAfter := billingErrorDetails(err)
		if retryAfter > 0 {
			c.Header("Retry-After", strconv.Itoa(retryAfter))
		}
		googleErrorWithType(c, status, code, "", message)
		return
	}

	inflightEstimate := tokenInflightEstimate(modelName, body)
	if isDecisions {
		inflightEstimate = service.InflightEstimateRequest{Model: modelName, BodyBytes: len(body), MaxTokens: 1, Kind: service.InflightEstimateToken}
	}
	inflightRelease, err := reserveInflightBalance(c, h.billingCacheService, h.gatewayService, apiKey, subscription, inflightEstimate)
	if err != nil {
		reqLog.Info("gemini.aistudio_endpoint.inflight_rejected", zap.Error(err))
		status, code, message, retryAfter := billingErrorDetails(err)
		if retryAfter > 0 {
			c.Header("Retry-After", strconv.Itoa(retryAfter))
		}
		googleErrorWithType(c, status, code, "", message)
		return
	}
	defer func() {
		if inflightRelease != nil {
			inflightRelease()
		}
	}()

	baseExclusions, err := h.gatewayService.GeminiAIStudioEndpointExclusions(c.Request.Context(), apiKey.GroupID, boundAccountID)
	if err != nil {
		reqLog.Warn("gemini.aistudio_endpoint.list_accounts_failed", zap.Error(err))
		googleError(c, http.StatusServiceUnavailable, "No available Gemini AI Studio accounts")
		return
	}

	maxSwitches := h.maxAccountSwitchesGemini
	if boundAccountID > 0 {
		maxSwitches = 0
	}
	fs := NewFailoverState(maxSwitches, boundAccountID > 0)

	for {
		excludedAccountIDs := mergeAccountIDSets(baseExclusions, fs.FailedAccountIDs)
		selection, err := h.gatewayService.SelectAccountWithLoadAwareness(c.Request.Context(), apiKey.GroupID, "", modelName, excludedAccountIDs, "", int64(0))
		if err == nil && (!service.IsGeminiAIStudioEndpointAccountType(selection.Account) || (boundAccountID > 0 && selection.Account.ID != boundAccountID)) {
			badID := selection.Account.ID
			if selection.Acquired && selection.ReleaseFunc != nil {
				selection.ReleaseFunc()
			}
			baseExclusions[badID] = struct{}{}
			continue
		}
		if err != nil {
			if failoverClientGone(c) {
				return
			}
			if len(fs.FailedAccountIDs) == 0 {
				cls := classifyNoAccountErrorFromGin(c, h.gatewayService, apiKey, modelName, modelName, service.PlatformGemini)
				if !cls.ModelNotFound {
					markOpsRoutingCapacityLimitedIfNoAvailable(c, err)
				}
				message := cls.Message
				if !cls.ModelNotFound {
					message = "No available Gemini AI Studio accounts: " + err.Error()
				}
				googleError(c, cls.Status, message)
				return
			}
			action := fs.HandleSelectionExhausted(c.Request.Context())
			switch action {
			case FailoverContinue:
				c.Request = c.Request.WithContext(service.WithSingleAccountRetry(c.Request.Context(), true, h.metadataBridgeEnabled()))
				continue
			case FailoverCanceled:
				failoverClientGone(c)
				return
			default:
				h.handleGeminiFailoverExhausted(c, fs.LastFailoverErr)
				return
			}
		}

		account := selection.Account
		accountReleaseFunc, ok := h.acquireGeminiEndpointAccountSlot(c, reqLog, geminiConcurrency, selection, stream, &streamStarted)
		if !ok {
			return
		}
		admissionCtx := service.ContextWithSelectionProfitGate(c.Request.Context(), selection)
		latest, vetoed, reason := h.gatewayService.GatewayProfitControlVetoLatest(admissionCtx, account)
		if vetoed || !service.IsGeminiAIStudioEndpointAccount(latest) {
			if accountReleaseFunc != nil {
				accountReleaseFunc()
			}
			if !vetoed {
				baseExclusions[account.ID] = struct{}{}
				continue
			}
			reqLog.Debug("gemini.aistudio_endpoint.profit_vetoed", zap.Int64("account_id", account.ID), zap.String("reason", reason))
			if fs.RecordProfitVeto(account.ID) == FailoverExhausted {
				markOpsRoutingCapacityLimited(c)
				googleError(c, http.StatusServiceUnavailable, profitVetoExhaustedMessage)
				return
			}
			continue
		}
		account = latest
		selection.Account = latest
		setOpsSelectedAccount(c, account.ID, account.Platform)
		accountReleaseFunc = wrapReleaseOnDone(c.Request.Context(), accountReleaseFunc)

		mappedModel := account.GetMappedModel(modelName)
		requestCtx := c.Request.Context()
		if fs.SwitchCount > 0 {
			requestCtx = service.WithAccountSwitchCount(requestCtx, fs.SwitchCount, h.metadataBridgeEnabled())
		}

		var (
			result             *service.ForwardResult
			interactionID      string
			boundInteractionID string
		)
		bindInteraction := func(id string) {
			if id == "" || boundInteractionID == id {
				return
			}
			bindCtx, cancel := context.WithTimeout(context.WithoutCancel(c.Request.Context()), geminiInteractionBindTimeout)
			defer cancel()
			allOK := true
			if bindErr := h.geminiCompatService.BindGeminiInteractionAccount(bindCtx, apiKey.GroupID, authSubject.UserID, id, account.ID); bindErr != nil {
				allOK = false
				reqLog.Warn("gemini.interaction.bind_account_failed", zap.String("interaction_id", id), zap.Int64("account_id", account.ID), zap.Error(bindErr))
			}
			var subID *int64
			if subscription != nil && subscription.ID > 0 {
				idCopy := subscription.ID
				subID = &idCopy
			}
			meta := service.GeminiInteractionMeta{
				RequestModel:       reqModel,
				UpstreamModel:      mappedModel,
				ChannelUsageFields: clientRequestedUsageFields(c, channelMapping, reqModel, mappedModel),
				PricingAtUnixNano:  pricingAt.UnixNano(),
				ForceCacheBilling:  fs.ForceCacheBilling,
				APIKeyID:           apiKey.ID,
				UserID:             authSubject.UserID,
				AccountID:          account.ID,
				SubscriptionID:     subID,
				RequestPayloadHash: service.HashUsageRequestPayload(body),
				InboundEndpoint:    GetInboundEndpoint(c),
				UpstreamEndpoint:   GetUpstreamEndpoint(c, account.Platform),
				QuotaPlatform:      service.QuotaPlatform(c.Request.Context(), apiKey),
				UserAgent:          c.GetHeader("User-Agent"),
				ClientIP:           ip.GetClientIP(c),
				SessionID:          service.ExtractClientSessionID(c),
			}
			if metaErr := h.geminiCompatService.SaveGeminiInteractionMeta(bindCtx, id, meta); metaErr != nil {
				allOK = false
				reqLog.Warn("gemini.interaction.save_meta_failed", zap.String("interaction_id", id), zap.Error(metaErr))
			}
			if schedErr := h.geminiCompatService.ScheduleGeminiInteractionPending(bindCtx, id, time.Now().Add(geminiInteractionSlowPollReschedule)); schedErr != nil {
				allOK = false
				reqLog.Warn("gemini.interaction.schedule_pending_failed", zap.String("interaction_id", id), zap.Error(schedErr))
			}
			if allOK {
				boundInteractionID = id
			}
		}
		if isDecisions {
			result, err = h.geminiCompatService.ForwardDecisions(requestCtx, c, account, modelName, mappedModel, body)
		} else {
			result, interactionID, err = h.geminiCompatService.ForwardInteractionCreate(requestCtx, c, account, modelName, mappedModel, stream, body, bindInteraction)
		}
		if accountReleaseFunc != nil {
			accountReleaseFunc()
		}
		if err != nil {
			var failoverErr *service.UpstreamFailoverError
			if errors.As(err, &failoverErr) {
				switch fs.HandleFailoverError(c.Request.Context(), h.gatewayService, account.ID, account.Platform, account.GetPoolModeRetryCount(), failoverErr) {
				case FailoverContinue:
					continue
				case FailoverExhausted:
					h.handleGeminiFailoverExhausted(c, fs.LastFailoverErr)
					return
				case FailoverCanceled:
					failoverClientGone(c)
					return
				}
			}
			failoverClientGone(c)
			reqLog.Error("gemini.aistudio_endpoint.forward_failed", zap.Int64("account_id", account.ID), zap.Error(err))
			if !isDecisions && boundInteractionID != "" && h.gatewayService != nil && h.geminiCompatService != nil {
				channelUsageFields := clientRequestedUsageFields(c, channelMapping, reqModel, mappedModel)
				snap := captureGeminiInteractionUsageSnapshot(c, apiKey, account, subscription, authSubject.UserID, pricingAt, reqModel, mappedModel, channelUsageFields, fs.ForceCacheBilling, body)
				handoffInflight := inflightRelease
				inflightRelease = nil
				go h.reconcileGeminiInteractionUsage(snap, boundInteractionID, handoffInflight)
			}
			return
		}

		pendingInteractionID := interactionID
		if pendingInteractionID == "" {
			pendingInteractionID = boundInteractionID
		}
		if !isDecisions && pendingInteractionID != "" {
			bindInteraction(pendingInteractionID)
		}

		if isDecisions || pendingInteractionID == "" || (result.InteractionTerminal && hasBillableGeminiResult(result)) {
			if !isDecisions && pendingInteractionID != "" && hasBillableGeminiResult(result) {
				claimCtx, cancel := context.WithTimeout(context.WithoutCancel(c.Request.Context()), geminiInteractionBindTimeout)
				claimed, claimErr := h.geminiCompatService.ClaimGeminiInteractionBilled(claimCtx, pendingInteractionID)
				cancel()
				if claimErr != nil || !claimed {
					if claimErr != nil {
						reqLog.Warn("gemini.interaction.claim_billed_failed", zap.String("interaction_id", pendingInteractionID), zap.Error(claimErr))
					}
					return
				}
			}
			channelUsageFields := clientRequestedUsageFields(c, channelMapping, reqModel, result.UpstreamModel)
			h.recordGeminiAIStudioUsage(c, apiKey, account, subscription, pricingAt, result, body, channelUsageFields, fs.ForceCacheBilling, component, pendingInteractionID)
			return
		}

		if result.InteractionTerminal && !result.Stream && !hasBillableGeminiResult(result) && !isDecisions && pendingInteractionID != "" {
			cleanCtx, cancelClean := context.WithTimeout(context.WithoutCancel(c.Request.Context()), geminiInteractionBindTimeout)
			_ = h.geminiCompatService.RemoveGeminiInteractionPending(cleanCtx, pendingInteractionID)
			cancelClean()
			return
		}

		if (!result.InteractionTerminal || (result.Stream && !hasBillableGeminiResult(result))) && h.gatewayService != nil && h.geminiCompatService != nil {
			channelUsageFields := clientRequestedUsageFields(c, channelMapping, reqModel, mappedModel)
			snap := captureGeminiInteractionUsageSnapshot(c, apiKey, account, subscription, authSubject.UserID, pricingAt, reqModel, mappedModel, channelUsageFields, fs.ForceCacheBilling, body)
			handoffInflight := inflightRelease
			inflightRelease = nil
			go h.reconcileGeminiInteractionUsage(snap, pendingInteractionID, handoffInflight)
		}
		return
	}
}

func (h *GatewayHandler) handleGeminiInteractionByID(c *gin.Context, method string, isCancel bool, stream bool) {
	apiKey, ok := middleware.GetAPIKeyFromContext(c)
	if !ok || apiKey == nil {
		googleError(c, http.StatusUnauthorized, "Invalid API key")
		return
	}
	authSubject, ok := middleware.GetAuthSubjectFromContext(c)
	if !ok {
		googleError(c, http.StatusInternalServerError, "User context not found")
		return
	}

	reqLog := requestLogger(
		c,
		"handler.gemini_v1beta.interactions",
		zap.Int64("user_id", authSubject.UserID),
		zap.Int64("api_key_id", apiKey.ID),
		zap.Any("group_id", apiKey.GroupID),
	)

	effectivePlatform := effectiveAPIKeyPlatform(c, apiKey)
	if effectivePlatform != service.PlatformGemini && effectivePlatform != service.PlatformComposite {
		googleError(c, http.StatusBadRequest, "API key group platform is not gemini")
		return
	}

	interactionID := strings.TrimSpace(c.Param("interactionID"))
	if !service.IsSafeGeminiInteractionID(interactionID) {
		googleError(c, http.StatusBadRequest, "Invalid interaction ID")
		return
	}

	boundAccountID, verified, err := h.geminiCompatService.ResolveGeminiInteractionAccount(
		c.Request.Context(),
		apiKey.GroupID,
		authSubject.UserID,
		interactionID,
	)
	if err != nil {
		reqLog.Warn("gemini.interaction.resolve_failed", zap.String("interaction_id", interactionID), zap.Error(err))
		googleError(c, http.StatusServiceUnavailable, "Interaction state temporarily unavailable")
		return
	}
	if !verified || boundAccountID <= 0 {
		googleError(c, http.StatusNotFound, "Interaction not found")
		return
	}

	setOpsRequestContext(c, "", stream)
	setOpsEndpointContext(c, "", int16(service.RequestTypeFromLegacy(stream, false)))
	pricingCtx, pricingAt := service.WithGatewayTokenRequestPricing(c.Request.Context())
	c.Request = c.Request.WithContext(pricingCtx)

	subscription, _ := middleware.GetSubscriptionFromContext(c)
	geminiConcurrency := NewConcurrencyHelper(h.concurrencyHelper.concurrencyService, SSEPingFormatNone, 0)
	streamStarted := false
	if h.errorPassthroughService != nil {
		service.BindErrorPassthroughService(c, h.errorPassthroughService)
	}
	userReleaseFunc, err := geminiConcurrency.AcquireUserSlotWithWait(c, authSubject.UserID, authSubject.Concurrency, stream, &streamStarted)
	if err != nil {
		reqLog.Warn("gemini.interaction.user_slot_acquire_failed", zap.Error(err))
		googleConcurrencyError(c, err, "user")
		return
	}
	userReleaseFunc = wrapReleaseOnDone(c.Request.Context(), userReleaseFunc)
	if userReleaseFunc != nil {
		defer userReleaseFunc()
	}

	baseExclusions, err := h.gatewayService.GeminiAIStudioEndpointExclusions(c.Request.Context(), apiKey.GroupID, boundAccountID)
	if err != nil {
		reqLog.Warn("gemini.interaction.list_accounts_failed", zap.Error(err))
		googleError(c, http.StatusServiceUnavailable, "No available Gemini AI Studio accounts")
		return
	}

	maxSwitches := h.maxAccountSwitchesGemini
	if boundAccountID > 0 {
		maxSwitches = 0
	}
	fs := NewFailoverState(maxSwitches, boundAccountID > 0)

	for {
		excludedAccountIDs := mergeAccountIDSets(baseExclusions, fs.FailedAccountIDs)
		selection, err := h.gatewayService.SelectAccountWithLoadAwareness(c.Request.Context(), apiKey.GroupID, "", "", excludedAccountIDs, "", int64(0))
		if err == nil && (!service.IsGeminiAIStudioEndpointAccountType(selection.Account) || (boundAccountID > 0 && selection.Account.ID != boundAccountID)) {
			badID := selection.Account.ID
			if selection.Acquired && selection.ReleaseFunc != nil {
				selection.ReleaseFunc()
			}
			baseExclusions[badID] = struct{}{}
			continue
		}
		if err != nil {
			if failoverClientGone(c) {
				return
			}
			if len(fs.FailedAccountIDs) == 0 {
				markOpsRoutingCapacityLimitedIfNoAvailable(c, err)
				googleError(c, http.StatusServiceUnavailable, "No available Gemini AI Studio accounts: "+err.Error())
				return
			}
			action := fs.HandleSelectionExhausted(c.Request.Context())
			switch action {
			case FailoverContinue:
				c.Request = c.Request.WithContext(service.WithSingleAccountRetry(c.Request.Context(), true, h.metadataBridgeEnabled()))
				continue
			case FailoverCanceled:
				failoverClientGone(c)
				return
			default:
				h.handleGeminiFailoverExhausted(c, fs.LastFailoverErr)
				return
			}
		}

		account := selection.Account
		accountReleaseFunc, ok := h.acquireGeminiEndpointAccountSlot(c, reqLog, geminiConcurrency, selection, stream, &streamStarted)
		if !ok {
			return
		}
		admissionCtx := service.ContextWithSelectionProfitGate(c.Request.Context(), selection)
		latest, vetoed, reason := h.gatewayService.GatewayProfitControlVetoLatest(admissionCtx, account)
		if vetoed || !service.IsGeminiAIStudioEndpointAccount(latest) {
			if accountReleaseFunc != nil {
				accountReleaseFunc()
			}
			if !vetoed {
				baseExclusions[account.ID] = struct{}{}
				continue
			}
			reqLog.Debug("gemini.interaction.profit_vetoed", zap.Int64("account_id", account.ID), zap.String("reason", reason))
			if fs.RecordProfitVeto(account.ID) == FailoverExhausted {
				markOpsRoutingCapacityLimited(c)
				googleError(c, http.StatusServiceUnavailable, profitVetoExhaustedMessage)
				return
			}
			continue
		}
		account = latest
		selection.Account = latest
		setOpsSelectedAccount(c, account.ID, account.Platform)
		accountReleaseFunc = wrapReleaseOnDone(c.Request.Context(), accountReleaseFunc)

		if method == http.MethodDelete {
			if ok := h.settleGeminiInteractionBeforeDelete(c, reqLog, account, apiKey, subscription, authSubject.UserID, pricingAt, fs.ForceCacheBilling, interactionID); !ok {
				if accountReleaseFunc != nil {
					accountReleaseFunc()
				}
				return
			}
		}

		result, observedModel, err := h.geminiCompatService.ForwardInteractionByID(
			c.Request.Context(),
			c,
			account,
			method,
			interactionID,
			isCancel,
			c.Request.URL.Query(),
			stream,
		)
		if accountReleaseFunc != nil {
			accountReleaseFunc()
		}
		if err != nil {
			var failoverErr *service.UpstreamFailoverError
			if errors.As(err, &failoverErr) {
				switch fs.HandleFailoverError(c.Request.Context(), h.gatewayService, account.ID, account.Platform, account.GetPoolModeRetryCount(), failoverErr) {
				case FailoverContinue:
					continue
				case FailoverExhausted:
					h.handleGeminiFailoverExhausted(c, fs.LastFailoverErr)
					return
				case FailoverCanceled:
					failoverClientGone(c)
					return
				}
			}
			failoverClientGone(c)
			reqLog.Error("gemini.interaction.forward_failed", zap.Int64("account_id", account.ID), zap.Error(err))
			return
		}

		if method == http.MethodDelete {
			settledCtx, cancelSettled := context.WithTimeout(context.WithoutCancel(c.Request.Context()), geminiInteractionBindTimeout)
			settled, settledErr := h.geminiCompatService.IsGeminiInteractionSettled(settledCtx, interactionID)
			cancelSettled()
			if settledErr == nil && settled {
				cleanCtx, cancelClean := context.WithTimeout(context.WithoutCancel(c.Request.Context()), geminiInteractionBindTimeout)
				_ = h.geminiCompatService.RemoveGeminiInteractionPending(cleanCtx, interactionID)
				cancelClean()
			}
			return
		}

		if (method == http.MethodGet || isCancel) && result.InteractionTerminal {
			if (result.Stream || isCancel) && !hasBillableGeminiResult(result) {
				return
			}
			_ = h.settleGeminiInteractionResponse(c, reqLog, account, apiKey, subscription, authSubject.UserID, pricingAt, fs.ForceCacheBilling, interactionID, result, observedModel, false)
		}
		return
	}
}

func (h *GatewayHandler) acquireGeminiEndpointAccountSlot(
	c *gin.Context,
	reqLog *zap.Logger,
	geminiConcurrency *ConcurrencyHelper,
	selection *service.AccountSelectionResult,
	stream bool,
	streamStarted *bool,
) (func(), bool) {
	if selection.Acquired {
		return selection.ReleaseFunc, true
	}
	if selection.WaitPlan == nil {
		markOpsRoutingCapacityLimited(c)
		googleError(c, http.StatusServiceUnavailable, "No available Gemini AI Studio accounts")
		return nil, false
	}
	account := selection.Account
	accountWaitCounted := false
	canWait, err := geminiConcurrency.IncrementAccountWaitCount(c.Request.Context(), account.ID, selection.WaitPlan.MaxWaiting)
	if err != nil {
		reqLog.Warn("gemini.aistudio_endpoint.account_wait_increment_failed", zap.Int64("account_id", account.ID), zap.Error(err))
	} else if !canWait {
		googleError(c, http.StatusTooManyRequests, "Too many pending requests, please retry later")
		return nil, false
	}
	if err == nil && canWait {
		accountWaitCounted = true
	}
	defer func() {
		if accountWaitCounted {
			geminiConcurrency.DecrementAccountWaitCount(c.Request.Context(), account.ID)
		}
	}()

	releaseFunc, err := geminiConcurrency.AcquireAccountSlotWithWaitTimeout(
		c,
		account.ID,
		selection.WaitPlan.MaxConcurrency,
		selection.WaitPlan.Timeout,
		stream,
		streamStarted,
	)
	if err != nil {
		reqLog.Warn("gemini.aistudio_endpoint.account_slot_acquire_failed", zap.Int64("account_id", account.ID), zap.Error(err))
		googleConcurrencyError(c, err, "account")
		return nil, false
	}
	return releaseFunc, true
}

type geminiInteractionUsageSnapshot struct {
	parentCtx          context.Context
	apiKey             *service.APIKey
	account            *service.Account
	subscription       *service.UserSubscription
	userID             int64
	pricingAt          time.Time
	requestModel       string
	upstreamModel      string
	channelUsageFields service.ChannelUsageFields
	forceCacheBilling  bool
	requestPayloadHash string
	inboundEndpoint    string
	upstreamEndpoint   string
	quotaPlatform      string
	userAgent          string
	clientIP           string
	sessionID          string
}

func captureGeminiInteractionUsageSnapshot(
	c *gin.Context,
	apiKey *service.APIKey,
	account *service.Account,
	subscription *service.UserSubscription,
	userID int64,
	pricingAt time.Time,
	requestModel string,
	upstreamModel string,
	channelUsageFields service.ChannelUsageFields,
	forceCacheBilling bool,
	body []byte,
) geminiInteractionUsageSnapshot {
	parentCtx := context.Background()
	if c != nil && c.Request != nil {
		parentCtx = context.WithoutCancel(c.Request.Context())
	}
	userAgent := ""
	clientIP := ""
	inboundEndpoint := ""
	upstreamEndpoint := ""
	sessionID := ""
	quotaPlatform := ""
	if c != nil {
		userAgent = c.GetHeader("User-Agent")
		clientIP = ip.GetClientIP(c)
		inboundEndpoint = GetInboundEndpoint(c)
		if account != nil {
			upstreamEndpoint = GetUpstreamEndpoint(c, account.Platform)
		}
		sessionID = service.ExtractClientSessionID(c)
		if c.Request != nil {
			quotaPlatform = service.QuotaPlatform(c.Request.Context(), apiKey)
		}
	}
	var payloadHash string
	if len(body) > 0 {
		payloadHash = service.HashUsageRequestPayload(body)
	}
	return geminiInteractionUsageSnapshot{
		parentCtx:          parentCtx,
		apiKey:             apiKey,
		account:            account,
		subscription:       subscription,
		userID:             userID,
		pricingAt:          pricingAt,
		requestModel:       requestModel,
		upstreamModel:      upstreamModel,
		channelUsageFields: channelUsageFields,
		forceCacheBilling:  forceCacheBilling,
		requestPayloadHash: payloadHash,
		inboundEndpoint:    inboundEndpoint,
		upstreamEndpoint:   upstreamEndpoint,
		quotaPlatform:      quotaPlatform,
		userAgent:          userAgent,
		clientIP:           clientIP,
		sessionID:          sessionID,
	}
}

func (h *GatewayHandler) releaseGeminiInteractionBilledBounded(parentCtx context.Context, interactionID string) {
	if h == nil || h.geminiCompatService == nil || interactionID == "" {
		return
	}
	baseCtx := context.Background()
	if parentCtx != nil {
		baseCtx = context.WithoutCancel(parentCtx)
	}
	releaseCtx, cancel := context.WithTimeout(baseCtx, geminiInteractionBindTimeout)
	defer cancel()
	_ = h.geminiCompatService.ReleaseGeminiInteractionBilled(releaseCtx, interactionID)
}

func (h *GatewayHandler) resolveGeminiInteractionSubscription(
	ctx context.Context,
	meta *service.GeminiInteractionMeta,
	currentSub *service.UserSubscription,
) (*service.UserSubscription, error) {
	if meta == nil {
		return currentSub, nil
	}
	if meta.SubscriptionID == nil {
		if meta.APIKeyID > 0 {
			return nil, nil
		}
		return currentSub, nil
	}
	if currentSub != nil && currentSub.ID == *meta.SubscriptionID {
		return currentSub, nil
	}
	if h == nil || h.gatewayService == nil || *meta.SubscriptionID <= 0 {
		return nil, nil
	}
	return h.gatewayService.GetUserSubscriptionByID(ctx, *meta.SubscriptionID)
}

func (h *GatewayHandler) settleGeminiInteractionBeforeDelete(
	c *gin.Context,
	reqLog *zap.Logger,
	account *service.Account,
	apiKey *service.APIKey,
	subscription *service.UserSubscription,
	userID int64,
	pricingAt time.Time,
	forceCacheBilling bool,
	interactionID string,
) bool {
	if h == nil || h.geminiCompatService == nil || interactionID == "" {
		return true
	}
	checkCtx, cancelCheck := context.WithTimeout(context.WithoutCancel(c.Request.Context()), geminiInteractionBindTimeout)
	settled, settledErr := h.geminiCompatService.IsGeminiInteractionSettled(checkCtx, interactionID)
	cancelCheck()
	if settledErr != nil {
		reqLog.Warn("gemini.interaction.pre_delete_settled_check_failed", zap.String("interaction_id", interactionID), zap.Error(settledErr))
		googleError(c, http.StatusServiceUnavailable, "Interaction billing temporarily unavailable")
		return false
	}
	if settled {
		return true
	}

	pollCtx, cancelPoll := context.WithTimeout(context.WithoutCancel(c.Request.Context()), geminiInteractionPollTimeout)
	snapRes, _, terminal, pollErr := h.geminiCompatService.FetchGeminiInteractionSnapshot(pollCtx, account, interactionID)
	cancelPoll()
	if pollErr != nil {
		if terminal {
			return true
		}
		reqLog.Warn("gemini.interaction.pre_delete_snapshot_failed", zap.String("interaction_id", interactionID), zap.Error(pollErr))
		googleError(c, http.StatusServiceUnavailable, "Interaction billing temporarily unavailable")
		return false
	}
	if !terminal {
		return true
	}
	if !h.settleGeminiInteractionResponse(c, reqLog, account, apiKey, subscription, userID, pricingAt, forceCacheBilling, interactionID, snapRes, "", true) {
		googleError(c, http.StatusServiceUnavailable, "Interaction billing temporarily unavailable")
		return false
	}
	return true
}

func (h *GatewayHandler) settleGeminiInteractionResponse(
	c *gin.Context,
	reqLog *zap.Logger,
	account *service.Account,
	apiKey *service.APIKey,
	subscription *service.UserSubscription,
	userID int64,
	pricingAt time.Time,
	forceCacheBilling bool,
	interactionID string,
	result *service.ForwardResult,
	observedModel string,
	waitSync bool,
) bool {
	if result == nil {
		return true
	}
	claimCtx, cancel := context.WithTimeout(context.WithoutCancel(c.Request.Context()), geminiInteractionBindTimeout)
	meta, hasMeta, metaErr := h.geminiCompatService.GetGeminiInteractionMeta(claimCtx, interactionID)
	cancel()
	if metaErr != nil {
		reqLog.Warn("gemini.interaction.get_meta_failed", zap.String("interaction_id", interactionID), zap.Error(metaErr))
		return false
	}
	service.ApplyGeminiInteractionMetaModels(result, observedModel, meta)
	if !hasBillableGeminiResult(result) {
		if !result.Stream {
			cleanCtx, cancelClean := context.WithTimeout(context.WithoutCancel(c.Request.Context()), geminiInteractionBindTimeout)
			_ = h.geminiCompatService.RemoveGeminiInteractionPending(cleanCtx, interactionID)
			cancelClean()
		}
		return true
	}

	claimCtx, cancelClaim := context.WithTimeout(context.WithoutCancel(c.Request.Context()), geminiInteractionBindTimeout)
	claimed, claimErr := h.geminiCompatService.ClaimGeminiInteractionBilled(claimCtx, interactionID)
	cancelClaim()
	if claimErr != nil {
		reqLog.Warn("gemini.interaction.claim_billed_failed", zap.String("interaction_id", interactionID), zap.Error(claimErr))
		return false
	}
	if !claimed {
		settledCtx, cancelSettled := context.WithTimeout(context.WithoutCancel(c.Request.Context()), geminiInteractionBindTimeout)
		settled, settledErr := h.geminiCompatService.IsGeminiInteractionSettled(settledCtx, interactionID)
		cancelSettled()
		return settledErr == nil && settled
	}

	billingAPIKey := apiKey
	billingSubscription := subscription
	resolveFailed := false
	if hasMeta && meta != nil {
		if meta.APIKeyID > 0 && meta.APIKeyID != apiKey.ID && h.apiKeyService != nil {
			resolveCtx, cancelResolve := context.WithTimeout(context.WithoutCancel(c.Request.Context()), geminiInteractionBindTimeout)
			originKey, keyErr := h.apiKeyService.GetByID(resolveCtx, meta.APIKeyID)
			cancelResolve()
			if keyErr == nil && originKey != nil {
				if originKey.User == nil && apiKey.User != nil && originKey.UserID == apiKey.User.ID {
					originKey.User = apiKey.User
				}
				billingAPIKey = originKey
			} else if !errors.Is(keyErr, service.ErrAPIKeyNotFound) {
				resolveFailed = true
				reqLog.Warn("gemini.interaction.get_origin_api_key_failed", zap.String("interaction_id", interactionID), zap.Int64("api_key_id", meta.APIKeyID), zap.Error(keyErr))
			}
		}
		if !resolveFailed {
			subCtx, cancelSub := context.WithTimeout(context.WithoutCancel(c.Request.Context()), geminiInteractionBindTimeout)
			resolvedSub, subErr := h.resolveGeminiInteractionSubscription(subCtx, meta, billingSubscription)
			cancelSub()
			if subErr != nil {
				resolveFailed = true
				reqLog.Warn("gemini.interaction.get_origin_subscription_failed", zap.String("interaction_id", interactionID), zap.Error(subErr))
			} else {
				billingSubscription = resolvedSub
			}
		}
	}
	if resolveFailed {
		h.releaseGeminiInteractionBilledBounded(c.Request.Context(), interactionID)
		return false
	}

	snap := captureGeminiInteractionUsageSnapshot(c, billingAPIKey, account, billingSubscription, userID, pricingAt, "", "", service.ChannelUsageFields{}, forceCacheBilling, nil)
	if hasMeta && meta != nil {
		if meta.PricingAtUnixNano > 0 {
			snap.pricingAt = time.Unix(0, meta.PricingAtUnixNano)
		}
		snap.forceCacheBilling = meta.ForceCacheBilling
		snap.channelUsageFields = meta.ChannelUsageFields
		snap.requestModel = meta.RequestModel
		snap.upstreamModel = meta.UpstreamModel
		if meta.UserID > 0 {
			snap.userID = meta.UserID
		}
		if meta.RequestPayloadHash != "" {
			snap.requestPayloadHash = meta.RequestPayloadHash
		}
		if meta.InboundEndpoint != "" {
			snap.inboundEndpoint = meta.InboundEndpoint
		}
		if meta.UpstreamEndpoint != "" {
			snap.upstreamEndpoint = meta.UpstreamEndpoint
		}
		if meta.QuotaPlatform != "" {
			snap.quotaPlatform = meta.QuotaPlatform
		}
		if meta.UserAgent != "" {
			snap.userAgent = meta.UserAgent
		}
		if meta.ClientIP != "" {
			snap.clientIP = meta.ClientIP
		}
		if meta.SessionID != "" {
			snap.sessionID = meta.SessionID
		}
	}
	if waitSync {
		return h.runMandatoryGeminiRecordUsage(snap, result, interactionID, "handler.gemini_v1beta.interactions") == nil
	}
	h.submitMandatoryUsageRecordTask(c.Request.Context(), h.buildGeminiRecordUsageTask(snap, result, interactionID, "handler.gemini_v1beta.interactions"))
	return true
}

func (h *GatewayHandler) recordGeminiInteractionUsageSync(
	ctx context.Context,
	snap geminiInteractionUsageSnapshot,
	result *service.ForwardResult,
	interactionIDForRetry string,
	component string,
) error {
	if interactionIDForRetry != "" && result != nil {
		result.RequestID = "gemini_interaction:" + interactionIDForRetry
	}
	var user *service.User
	if snap.apiKey != nil {
		user = snap.apiKey.User
	}
	if err := h.gatewayService.RecordUsage(ctx, &service.RecordUsageInput{
		Result:             result,
		QuotaPlatform:      snap.quotaPlatform,
		APIKey:             snap.apiKey,
		User:               user,
		Account:            snap.account,
		Subscription:       snap.subscription,
		PricingAt:          snap.pricingAt,
		InboundEndpoint:    snap.inboundEndpoint,
		UpstreamEndpoint:   snap.upstreamEndpoint,
		UserAgent:          snap.userAgent,
		IPAddress:          snap.clientIP,
		RequestPayloadHash: snap.requestPayloadHash,
		ForceCacheBilling:  snap.forceCacheBilling,
		APIKeyService:      h.apiKeyService,
		SessionID:          snap.sessionID,
		ChannelUsageFields: snap.channelUsageFields,
	}); err != nil {
		if interactionIDForRetry != "" && h.geminiCompatService != nil {
			_ = h.geminiCompatService.ReleaseGeminiInteractionBilled(ctx, interactionIDForRetry)
		}
		var apiKeyID int64
		var groupID *int64
		var accountID int64
		if snap.apiKey != nil {
			apiKeyID = snap.apiKey.ID
			groupID = snap.apiKey.GroupID
		}
		if snap.account != nil {
			accountID = snap.account.ID
		}
		logger.L().With(
			zap.String("component", component),
			zap.Int64("user_id", snap.userID),
			zap.Int64("api_key_id", apiKeyID),
			zap.Any("group_id", groupID),
			zap.String("model", result.Model),
			zap.Int64("account_id", accountID),
		).Error("gemini.aistudio_endpoint.record_usage_failed", zap.Error(err))
		return err
	}
	if interactionIDForRetry != "" && h.geminiCompatService != nil {
		_ = h.geminiCompatService.MarkGeminiInteractionSettled(ctx, interactionIDForRetry)
	}
	return nil
}

func (h *GatewayHandler) buildGeminiRecordUsageTask(
	snap geminiInteractionUsageSnapshot,
	result *service.ForwardResult,
	interactionIDForRetry string,
	component string,
) service.UsageRecordTask {
	return func(ctx context.Context) {
		_ = h.recordGeminiInteractionUsageSync(ctx, snap, result, interactionIDForRetry, component)
	}
}

func (h *GatewayHandler) runMandatoryGeminiRecordUsage(
	snap geminiInteractionUsageSnapshot,
	result *service.ForwardResult,
	interactionIDForRetry string,
	component string,
) error {
	doneCh := make(chan error, 1)
	h.submitMandatoryUsageRecordTask(snap.parentCtx, func(ctx context.Context) {
		completed := false
		defer func() {
			if !completed {
				doneCh <- errors.New("gemini interaction usage task aborted before completion")
			}
		}()
		err := h.recordGeminiInteractionUsageSync(ctx, snap, result, interactionIDForRetry, component)
		completed = true
		doneCh <- err
	})
	waitTimeout := geminiInteractionMandatoryWaitTimeout
	if waitTimeout <= 0 {
		waitTimeout = 20 * time.Second
	}
	timer := time.NewTimer(waitTimeout)
	defer timer.Stop()
	select {
	case err := <-doneCh:
		return err
	case <-timer.C:
		return errors.New("timed out waiting for mandatory usage record task")
	}
}

func (h *GatewayHandler) recordGeminiAIStudioUsage(
	c *gin.Context,
	apiKey *service.APIKey,
	account *service.Account,
	subscription *service.UserSubscription,
	pricingAt time.Time,
	result *service.ForwardResult,
	body []byte,
	channelUsageFields service.ChannelUsageFields,
	forceCacheBilling bool,
	component string,
	interactionIDForRetry string,
) {
	if h.gatewayService == nil || result == nil {
		return
	}
	authSubject, _ := middleware.GetAuthSubjectFromContext(c)
	snap := captureGeminiInteractionUsageSnapshot(c, apiKey, account, subscription, authSubject.UserID, pricingAt, result.Model, result.UpstreamModel, channelUsageFields, forceCacheBilling, body)
	task := h.buildGeminiRecordUsageTask(snap, result, interactionIDForRetry, component)
	if interactionIDForRetry != "" {
		h.submitMandatoryUsageRecordTask(c.Request.Context(), task)
		return
	}
	h.submitUsageRecordTask(c.Request.Context(), task)
}

func (h *GatewayHandler) geminiReconcileStopCh() <-chan struct{} {
	if h == nil || h.geminiReconcileStop == nil {
		return nil
	}
	return h.geminiReconcileStop
}

func (h *GatewayHandler) startGeminiInteractionPendingSweeper() {
	if h == nil || h.geminiCompatService == nil || h.gatewayService == nil {
		return
	}
	h.geminiReconcileOnce.Do(func() {
		if h.geminiReconcileStop == nil {
			h.geminiReconcileStop = make(chan struct{})
		}
		go h.runGeminiInteractionPendingSweeper()
	})
}

// StopGeminiInteractionPendingSweeper stops the background delayed-interaction billing sweeper.
func (h *GatewayHandler) StopGeminiInteractionPendingSweeper() {
	if h == nil {
		return
	}
	h.geminiReconcileStopOnce.Do(func() {
		if h.geminiReconcileStop != nil {
			close(h.geminiReconcileStop)
		}
	})
}

func (h *GatewayHandler) runGeminiInteractionPendingSweeper() {
	interval := geminiInteractionPendingSweepEvery
	if interval <= 0 {
		interval = geminiInteractionPendingSweepInterval
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	// Run an initial pass shortly after startup to recover interactions persisted before restart.
	h.ReconcilePendingGeminiInteractions(context.Background())

	for {
		select {
		case <-h.geminiReconcileStop:
			return
		case <-ticker.C:
			h.ReconcilePendingGeminiInteractions(context.Background())
		}
	}
}

func (h *GatewayHandler) ensureGeminiInteractionPendingEnqueued(
	snap geminiInteractionUsageSnapshot,
	interactionID string,
	nextCheckAt time.Time,
) {
	if h == nil || h.geminiCompatService == nil || interactionID == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), geminiInteractionBindTimeout)
	defer cancel()

	_, hasMeta, metaErr := h.geminiCompatService.GetGeminiInteractionMeta(ctx, interactionID)
	if metaErr == nil && !hasMeta {
		var (
			apiKeyID int64
			acctID   int64
			subID    *int64
		)
		if snap.apiKey != nil {
			apiKeyID = snap.apiKey.ID
		}
		if snap.account != nil {
			acctID = snap.account.ID
		}
		if snap.subscription != nil && snap.subscription.ID > 0 {
			idCopy := snap.subscription.ID
			subID = &idCopy
		}
		reqModel := snap.requestModel
		if reqModel == "" {
			reqModel = snap.channelUsageFields.OriginalModel
		}
		upModel := snap.upstreamModel
		if upModel == "" {
			upModel = snap.channelUsageFields.ChannelMappedModel
		}
		_ = h.geminiCompatService.SaveGeminiInteractionMeta(ctx, interactionID, service.GeminiInteractionMeta{
			RequestModel:       reqModel,
			UpstreamModel:      upModel,
			ChannelUsageFields: snap.channelUsageFields,
			PricingAtUnixNano:  snap.pricingAt.UnixNano(),
			ForceCacheBilling:  snap.forceCacheBilling,
			APIKeyID:           apiKeyID,
			UserID:             snap.userID,
			AccountID:          acctID,
			SubscriptionID:     subID,
			RequestPayloadHash: snap.requestPayloadHash,
			InboundEndpoint:    snap.inboundEndpoint,
			UpstreamEndpoint:   snap.upstreamEndpoint,
			QuotaPlatform:      snap.quotaPlatform,
			UserAgent:          snap.userAgent,
			ClientIP:           snap.clientIP,
			SessionID:          snap.sessionID,
		})
	}
	_ = h.geminiCompatService.ScheduleGeminiInteractionPending(ctx, interactionID, nextCheckAt)
}

func (h *GatewayHandler) reconcileGeminiInteractionUsage(
	snap geminiInteractionUsageSnapshot,
	interactionID string,
	inflightDone func(),
) {
	defer func() {
		if inflightDone != nil {
			inflightDone()
		}
	}()
	if h == nil || h.geminiCompatService == nil || h.gatewayService == nil || snap.account == nil || snap.apiKey == nil || interactionID == "" {
		return
	}
	h.startGeminiInteractionPendingSweeper()
	h.ensureGeminiInteractionPendingEnqueued(snap, interactionID, time.Now().Add(geminiInteractionSlowPollReschedule))

	maxWait := geminiInteractionReconcileMaxWait
	if maxWait <= 0 {
		maxWait = defaultGeminiInteractionReconcileMaxWait
	}
	deadline := time.Now().Add(maxWait)
	intervals := geminiInteractionReconcileIntervals
	if len(intervals) == 0 {
		intervals = []time.Duration{5 * time.Second}
	}

	for attempt := 0; ; attempt++ {
		settledCtx, cancelSettled := context.WithTimeout(context.Background(), geminiInteractionBindTimeout)
		settled, settledErr := h.geminiCompatService.IsGeminiInteractionSettled(settledCtx, interactionID)
		cancelSettled()
		if settledErr == nil && settled {
			removeCtx, cancelRemove := context.WithTimeout(context.Background(), geminiInteractionBindTimeout)
			_ = h.geminiCompatService.RemoveGeminiInteractionPending(removeCtx, interactionID)
			cancelRemove()
			return
		}

		pollCtx, cancelPoll := context.WithTimeout(context.Background(), geminiInteractionPollTimeout)
		res, _, terminal, err := h.geminiCompatService.FetchGeminiInteractionSnapshot(pollCtx, snap.account, interactionID)
		cancelPoll()
		if err == nil && terminal {
			service.ApplyGeminiInteractionMetaModels(res, "", &service.GeminiInteractionMeta{
				RequestModel:  snap.requestModel,
				UpstreamModel: snap.upstreamModel,
			})
			if !hasBillableGeminiResult(res) {
				removeCtx, cancelRemove := context.WithTimeout(context.Background(), geminiInteractionBindTimeout)
				_ = h.geminiCompatService.RemoveGeminiInteractionPending(removeCtx, interactionID)
				cancelRemove()
				return
			}
			claimCtx, cancelClaim := context.WithTimeout(context.Background(), geminiInteractionBindTimeout)
			claimed, claimErr := h.geminiCompatService.ClaimGeminiInteractionBilled(claimCtx, interactionID)
			cancelClaim()
			if claimErr == nil && claimed {
				if recordErr := h.runMandatoryGeminiRecordUsage(snap, res, interactionID, "handler.gemini_v1beta.interactions.reconcile"); recordErr == nil {
					return
				}
			} else if claimErr != nil {
				logger.L().Warn("gemini.interaction.reconcile_claim_failed", zap.String("interaction_id", interactionID), zap.Error(claimErr))
			}
		}

		remaining := time.Until(deadline)
		if remaining <= 0 {
			// Retain durable pending tracking so slow sweeper keeps polling/retrying past fast-reconcile deadline.
			h.ensureGeminiInteractionPendingEnqueued(snap, interactionID, time.Now().Add(geminiInteractionSlowPollReschedule))
			return
		}

		idx := attempt
		if idx >= len(intervals) {
			idx = len(intervals) - 1
		}
		wait := intervals[idx]
		if wait > remaining {
			wait = remaining
		}
		timer := time.NewTimer(wait)
		select {
		case <-timer.C:
		case <-h.geminiReconcileStopCh():
			timer.Stop()
			h.ensureGeminiInteractionPendingEnqueued(snap, interactionID, time.Now().Add(geminiInteractionSlowPollReschedule))
			return
		}
	}
}

// ReconcilePendingGeminiInteractions sweeps overdue durable pending interactions
// (interactions exceeding the 20m fast poller, transient RecordUsage failures, or server restarts).
func (h *GatewayHandler) ReconcilePendingGeminiInteractions(ctx context.Context) {
	if h == nil || h.geminiCompatService == nil || h.gatewayService == nil {
		return
	}
	if ctx == nil {
		ctx = context.Background()
	}

	listCtx, cancelList := context.WithTimeout(ctx, geminiInteractionBindTimeout)
	ids, err := h.geminiCompatService.ListDueGeminiInteractionPending(listCtx, time.Now(), geminiInteractionPendingSweepBatchSize)
	cancelList()
	if err != nil || len(ids) == 0 {
		return
	}

	rescheduleInterval := geminiInteractionSlowPollReschedule
	if rescheduleInterval <= 0 {
		rescheduleInterval = geminiInteractionSlowReconcileInterval
	}

	for _, interactionID := range ids {
		settledCtx, cancelSettled := context.WithTimeout(ctx, geminiInteractionBindTimeout)
		settled, settledErr := h.geminiCompatService.IsGeminiInteractionSettled(settledCtx, interactionID)
		cancelSettled()
		if settledErr == nil && settled {
			remCtx, cancelRem := context.WithTimeout(ctx, geminiInteractionBindTimeout)
			_ = h.geminiCompatService.RemoveGeminiInteractionPending(remCtx, interactionID)
			cancelRem()
			continue
		}

		metaCtx, cancelMeta := context.WithTimeout(ctx, geminiInteractionBindTimeout)
		meta, hasMeta, metaErr := h.geminiCompatService.GetGeminiInteractionMeta(metaCtx, interactionID)
		cancelMeta()
		if metaErr != nil {
			schedCtx, cancelSched := context.WithTimeout(ctx, geminiInteractionBindTimeout)
			_ = h.geminiCompatService.ScheduleGeminiInteractionPending(schedCtx, interactionID, time.Now().Add(rescheduleInterval))
			cancelSched()
			continue
		}
		if !hasMeta || meta == nil || meta.AccountID <= 0 ||
			(meta.PricingAtUnixNano > 0 && time.Since(time.Unix(0, meta.PricingAtUnixNano)) > geminiInteractionPendingMaxAge) {
			remCtx, cancelRem := context.WithTimeout(ctx, geminiInteractionBindTimeout)
			_ = h.geminiCompatService.RemoveGeminiInteractionPending(remCtx, interactionID)
			cancelRem()
			continue
		}

		// Push nextCheckAt forward before polling upstream to avoid duplicate work across sweeps.
		schedCtx, cancelSched := context.WithTimeout(ctx, geminiInteractionBindTimeout)
		_ = h.geminiCompatService.ScheduleGeminiInteractionPending(schedCtx, interactionID, time.Now().Add(rescheduleInterval))
		cancelSched()

		acctCtx, cancelAcct := context.WithTimeout(ctx, geminiInteractionBindTimeout)
		account, acctErr := h.geminiCompatService.GetAccountByID(acctCtx, meta.AccountID)
		cancelAcct()
		if acctErr != nil || account == nil {
			continue
		}

		pollCtx, cancelPoll := context.WithTimeout(ctx, geminiInteractionPollTimeout)
		res, _, terminal, pollErr := h.geminiCompatService.FetchGeminiInteractionSnapshot(pollCtx, account, interactionID)
		cancelPoll()
		if pollErr != nil {
			if terminal {
				remCtx, cancelRem := context.WithTimeout(ctx, geminiInteractionBindTimeout)
				_ = h.geminiCompatService.RemoveGeminiInteractionPending(remCtx, interactionID)
				cancelRem()
			}
			continue
		}
		if !terminal {
			continue
		}

		service.ApplyGeminiInteractionMetaModels(res, "", meta)
		if !hasBillableGeminiResult(res) {
			remCtx, cancelRem := context.WithTimeout(ctx, geminiInteractionBindTimeout)
			_ = h.geminiCompatService.RemoveGeminiInteractionPending(remCtx, interactionID)
			cancelRem()
			continue
		}

		if h.apiKeyService == nil || meta.APIKeyID <= 0 {
			continue
		}
		keyCtx, cancelKey := context.WithTimeout(ctx, geminiInteractionBindTimeout)
		billingAPIKey, keyErr := h.apiKeyService.GetByID(keyCtx, meta.APIKeyID)
		cancelKey()
		if keyErr != nil || billingAPIKey == nil {
			if errors.Is(keyErr, service.ErrAPIKeyNotFound) {
				remCtx, cancelRem := context.WithTimeout(ctx, geminiInteractionBindTimeout)
				_ = h.geminiCompatService.RemoveGeminiInteractionPending(remCtx, interactionID)
				cancelRem()
			}
			continue
		}

		subCtx, cancelSub := context.WithTimeout(ctx, geminiInteractionBindTimeout)
		billingSubscription, subErr := h.resolveGeminiInteractionSubscription(subCtx, meta, nil)
		cancelSub()
		if subErr != nil {
			continue
		}

		claimCtx, cancelClaim := context.WithTimeout(ctx, geminiInteractionBindTimeout)
		claimed, claimErr := h.geminiCompatService.ClaimGeminiInteractionBilled(claimCtx, interactionID)
		cancelClaim()
		if claimErr != nil || !claimed {
			continue
		}

		pricingAt := time.Now()
		if meta.PricingAtUnixNano > 0 {
			pricingAt = time.Unix(0, meta.PricingAtUnixNano)
		}
		userID := meta.UserID
		if userID == 0 {
			userID = billingAPIKey.UserID
		}
		snap := geminiInteractionUsageSnapshot{
			parentCtx:          context.Background(),
			apiKey:             billingAPIKey,
			account:            account,
			subscription:       billingSubscription,
			userID:             userID,
			pricingAt:          pricingAt,
			requestModel:       meta.RequestModel,
			upstreamModel:      meta.UpstreamModel,
			channelUsageFields: meta.ChannelUsageFields,
			forceCacheBilling:  meta.ForceCacheBilling,
			requestPayloadHash: meta.RequestPayloadHash,
			inboundEndpoint:    meta.InboundEndpoint,
			upstreamEndpoint:   meta.UpstreamEndpoint,
			quotaPlatform:      meta.QuotaPlatform,
			userAgent:          meta.UserAgent,
			clientIP:           meta.ClientIP,
			sessionID:          meta.SessionID,
		}
		_ = h.runMandatoryGeminiRecordUsage(snap, res, interactionID, "handler.gemini_v1beta.interactions.sweep")
	}
}

func hasBillableGeminiResult(result *service.ForwardResult) bool {
	if result == nil {
		return false
	}
	return hasBillableGeminiUsage(result.Usage) || result.ImageCount > 0
}

func hasBillableGeminiUsage(u service.ClaudeUsage) bool {
	return u.InputTokens > 0 || u.OutputTokens > 0 || u.CacheReadInputTokens > 0 || u.ImageOutputTokens > 0
}
