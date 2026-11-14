package handler

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	middleware2 "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	coderws "github.com/coder/websocket"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"go.uber.org/zap"
)

const (
	geminiLiveTokenMetaCtxKey = "gemini_live_token_meta"
	geminiLiveRawTokenCtxKey  = "gemini_live_raw_token"
	geminiLiveSetupTimeout    = 15 * time.Second
	geminiLiveDialTimeout     = 20 * time.Second
)

// ExtractGeminiLiveConstrainedAccessToken extracts an ephemeral access_token from query string or Authorization header.
func ExtractGeminiLiveConstrainedAccessToken(c *gin.Context) string {
	if c == nil || c.Request == nil {
		return ""
	}
	if tok := strings.TrimSpace(c.Query("access_token")); tok != "" {
		return tok
	}
	auth := strings.TrimSpace(c.GetHeader("Authorization"))
	if strings.HasPrefix(auth, "Token ") {
		return strings.TrimSpace(strings.TrimPrefix(auth, "Token "))
	}
	return ""
}

// StashGeminiLiveConstrainedAuth attaches resolved ephemeral token metadata onto the request context.
func StashGeminiLiveConstrainedAuth(c *gin.Context, meta *service.GeminiAuthTokenMeta, rawToken string) {
	if c == nil {
		return
	}
	if meta != nil {
		c.Set(geminiLiveTokenMetaCtxKey, meta)
	}
	if rawToken != "" {
		c.Set(geminiLiveRawTokenCtxKey, rawToken)
	}
}

func getStashedGeminiLiveConstrainedAuth(c *gin.Context) (*service.GeminiAuthTokenMeta, string) {
	if c == nil {
		return nil, ""
	}
	var meta *service.GeminiAuthTokenMeta
	if v, ok := c.Get(geminiLiveTokenMetaCtxKey); ok {
		meta, _ = v.(*service.GeminiAuthTokenMeta)
	}
	var raw string
	if v, ok := c.Get(geminiLiveRawTokenCtxKey); ok {
		raw, _ = v.(string)
	}
	return meta, raw
}

// GeminiLiveWS handles full-duplex WebSocket streaming on BidiGenerateContent and /v1beta/live.
func (h *GatewayHandler) GeminiLiveWS(c *gin.Context) {
	h.handleGeminiLiveWS(c, false)
}

// GeminiLiveWSConstrained handles full-duplex WebSocket streaming on BidiGenerateContentConstrained
// (supporting both standard gateway API keys and short-lived auth_tokens/... ephemeral tokens).
func (h *GatewayHandler) GeminiLiveWSConstrained(c *gin.Context) {
	h.handleGeminiLiveWS(c, true)
}

func isExpectedGeminiLiveWSClose(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) {
		return true
	}
	switch coderws.CloseStatus(err) {
	case coderws.StatusNormalClosure, coderws.StatusGoingAway,
		coderws.StatusNoStatusRcvd, coderws.StatusAbnormalClosure:
		return true
	default:
		return false
	}
}

func (h *GatewayHandler) handleGeminiLiveWS(c *gin.Context, constrained bool) {
	apiKey, ok := middleware2.GetAPIKeyFromContext(c)
	if !ok || apiKey == nil {
		googleError(c, http.StatusUnauthorized, "Invalid API key")
		return
	}
	subject, ok := middleware2.GetAuthSubjectFromContext(c)
	if !ok {
		googleError(c, http.StatusInternalServerError, "User context not found")
		return
	}
	reqLog := requestLogger(
		c,
		"handler.gemini_live_ws",
		zap.Int64("user_id", subject.UserID),
		zap.Int64("api_key_id", apiKey.ID),
		zap.Any("group_id", apiKey.GroupID),
		zap.Bool("constrained", constrained),
	)
	if h.geminiCompatService == nil || h.gatewayService == nil {
		googleError(c, http.StatusServiceUnavailable, "Gemini Live service unavailable")
		return
	}

	tokenMeta, rawEphemeralToken := getStashedGeminiLiveConstrainedAuth(c)
	subscription := h.resolveGeminiExtSubscription(c, apiKey)
	if h.billingCacheService != nil && apiKey.User != nil {
		if err := h.billingCacheService.CheckBillingEligibility(c.Request.Context(), apiKey.User, apiKey, apiKey.Group, subscription, service.PlatformGemini); err != nil {
			status, _, message, _ := billingErrorDetails(err)
			googleError(c, status, message)
			return
		}
	}
	if tokenMeta != nil && tokenMeta.Model != "" {
		if !h.checkGeminiGroupModelAllowlist(c, apiKey, tokenMeta.Model) {
			return
		}
	}

	userRelease, err := h.concurrencyHelper.AcquireUserSlotWithWait(c, subject.UserID, subject.Concurrency, true, nil)
	if err != nil {
		h.handleConcurrencyError(c, err, "user", false)
		return
	}
	defer userRelease()

	clientConn, err := coderws.Accept(c.Writer, c.Request, &coderws.AcceptOptions{
		CompressionMode: coderws.CompressionContextTakeover,
	})
	if err != nil {
		reqLog.Warn("gemini.live_ws.accept_failed", zap.Error(err))
		return
	}
	// 默认放宽单帧读取上限，兼容实时视频/音频分片。
	clientConn.SetReadLimit(16 << 20)
	defer func() { _ = clientConn.CloseNow() }()

	setupCtx, cancelSetup := context.WithTimeout(c.Request.Context(), geminiLiveSetupTimeout)
	firstMsgType, firstFrame, readErr := clientConn.Read(setupCtx)
	cancelSetup()
	if readErr != nil {
		_ = clientConn.Close(coderws.StatusPolicyViolation, "missing setup message")
		return
	}
	if firstMsgType != coderws.MessageText && firstMsgType != coderws.MessageBinary {
		_ = clientConn.Close(coderws.StatusUnsupportedData, "setup frame must be text or binary JSON")
		return
	}

	reqModel := service.ParseGeminiLiveSetupModel(firstFrame)
	if reqModel == "" && tokenMeta != nil {
		reqModel = tokenMeta.Model
	}
	if reqModel == "" {
		reqModel = "gemini-3.8-live"
	}

	if apiKey.Group != nil && apiKey.Group.ModelAllowlistEnabled() && !apiKey.Group.ModelAllowlist.Allows(reqModel) {
		_ = clientConn.Close(coderws.StatusPolicyViolation, fmt.Sprintf("Model '%s' is not allowed by group model allowlist", reqModel))
		return
	}

	if auditBody := service.BuildGeminiLiveAuditJSON(firstFrame); len(auditBody) > 0 {
		if decision := h.checkSecurityAuditStage(c, reqLog, apiKey, subject, "gemini", reqModel, auditBody, "first_turn"); decision != nil && !decision.AllowNextStage {
			msg := decision.ClientMessage
			if msg == "" {
				msg = "request blocked by security policy"
			}
			_ = clientConn.Close(coderws.StatusPolicyViolation, msg)
			return
		}
	}

	if apiKey.Group != nil && apiKey.Group.Platform == service.PlatformComposite {
		decision, routeErr := h.gatewayService.ResolveCompositeRoute(c.Request.Context(), apiKey.Group.ID, reqModel, service.CompositeRouteEndpointGemini)
		if routeErr != nil {
			_ = clientConn.Close(coderws.StatusInternalError, "failed to resolve composite model route")
			return
		}
		if decision.Matched {
			if decision.TargetPlatform != service.PlatformGemini {
				_ = clientConn.Close(coderws.StatusPolicyViolation, "Model is not routed to Gemini platform in this composite group")
				return
			}
			c.Request = c.Request.WithContext(service.WithCompositeRouteDecision(c.Request.Context(), decision))
		}
	}
	ensureCompositeTargetPlatform(c, apiKey, reqModel)
	if !compositeTargetPlatformAllowed(c, apiKey, reqModel, service.PlatformGemini) {
		_ = clientConn.Close(coderws.StatusPolicyViolation, "Model is not routed to Gemini platform in this composite group")
		return
	}
	routeModel := reqModel
	if upstreamModel, ok := service.ResolvedUpstreamModelFromContext(c.Request.Context()); ok && upstreamModel != "" {
		routeModel = upstreamModel
	}
	channelMapping, _ := h.gatewayService.ResolveChannelMappingAndRestrict(c.Request.Context(), apiKey.GroupID, routeModel)
	if channelMapping.Mapped && channelMapping.MappedModel != "" {
		routeModel = channelMapping.MappedModel
	}

	inflightRelease, inflightErr := reserveInflightBalance(c, h.billingCacheService, h.gatewayService, apiKey, subscription, tokenInflightEstimate(routeModel, firstFrame))
	if inflightErr != nil {
		_, _, message, _ := billingErrorDetails(inflightErr)
		_ = clientConn.Close(coderws.StatusPolicyViolation, message)
		return
	}
	defer inflightRelease()

	var boundAccountID int64
	if tokenMeta != nil && tokenMeta.AccountID > 0 {
		boundAccountID = tokenMeta.AccountID
	}

	geminiConcurrency := NewConcurrencyHelper(h.concurrencyHelper.concurrencyService, SSEPingFormatNone, 0)
	excludedIDs, _ := h.gatewayService.GeminiAIStudioEndpointExclusions(c.Request.Context(), apiKey.GroupID, boundAccountID)
	if excludedIDs == nil {
		excludedIDs = make(map[int64]struct{})
	}
	maxSwitches := h.maxAccountSwitchesGemini
	if boundAccountID > 0 {
		maxSwitches = 0
	}
	fs := NewFailoverState(maxSwitches, boundAccountID > 0)
	sessionHash := "gemini:live:user:" + strconv.FormatInt(subject.UserID, 10)

	var (
		account        *service.Account
		accountRelease func()
		upstreamConn   service.GeminiLiveWSConn
		mappedModel    string
	)

	upstreamEphemeralToken := rawEphemeralToken
	if tokenMeta != nil && tokenMeta.UpstreamToken != "" {
		upstreamEphemeralToken = tokenMeta.UpstreamToken
	}

	for {
		var selection *service.AccountSelectionResult
		var selErr error
		if boundAccountID > 0 {
			pinnedAcc, ok := h.loadValidatedPinnedGeminiAccount(c.Request.Context(), apiKey, boundAccountID)
			if !ok {
				_ = clientConn.Close(coderws.StatusTryAgainLater, "pinned upstream account unavailable")
				return
			}
			account = pinnedAcc
		} else {
			selection, selErr = h.gatewayService.SelectAccountWithLoadAwareness(
				c.Request.Context(), apiKey.GroupID, sessionHash, routeModel, excludedIDs, "", subject.UserID,
			)
			if selErr != nil || selection == nil || selection.Account == nil {
				switch fs.HandleSelectionExhausted(c.Request.Context()) {
				case FailoverContinue:
					continue
				default:
					_ = clientConn.Close(coderws.StatusTryAgainLater, "no available Gemini Live accounts")
					return
				}
			}
			account = selection.Account
			if !service.IsGeminiAIStudioEndpointAccountType(account) {
				if selection.Acquired && selection.ReleaseFunc != nil {
					selection.ReleaseFunc()
				}
				excludedIDs[account.ID] = struct{}{}
				continue
			}
		}

		latest, vetoed, _ := h.gatewayService.GatewayProfitControlVetoLatest(c.Request.Context(), account)
		if vetoed || !service.IsGeminiAIStudioEndpointAccount(latest) {
			if selection != nil && selection.Acquired && selection.ReleaseFunc != nil {
				selection.ReleaseFunc()
			}
			excludedIDs[account.ID] = struct{}{}
			if boundAccountID > 0 || fs.RecordProfitVeto(account.ID) == FailoverExhausted {
				_ = clientConn.Close(coderws.StatusTryAgainLater, "no available Gemini Live accounts")
				return
			}
			continue
		}
		account = latest

		var rel func()
		if selection != nil {
			if selection.Acquired {
				rel = selection.ReleaseFunc
			} else {
				tryRel, acquired, slotErr := geminiConcurrency.TryAcquireAccountSlot(c.Request.Context(), account.ID, account.Concurrency)
				if slotErr != nil || !acquired {
					excludedIDs[account.ID] = struct{}{}
					if boundAccountID > 0 {
						_ = clientConn.Close(coderws.StatusTryAgainLater, "account concurrency limit reached")
						return
					}
					continue
				}
				rel = tryRel
			}
		} else {
			tryRel, acquired, slotErr := geminiConcurrency.TryAcquireAccountSlot(c.Request.Context(), account.ID, account.Concurrency)
			if slotErr != nil || !acquired {
				_ = clientConn.Close(coderws.StatusTryAgainLater, "pinned account concurrency limit reached")
				return
			}
			rel = tryRel
		}

		mappedModel = account.GetMappedModel(routeModel)
		dialCtx, cancelDial := context.WithTimeout(c.Request.Context(), geminiLiveDialTimeout)
		upConn, statusCode, _, dialErr := h.geminiCompatService.DialGeminiLiveWS(dialCtx, account, constrained, c.Request.URL.Query(), upstreamEphemeralToken)
		cancelDial()
		if dialErr != nil {
			rel()
			reqLog.Warn("gemini.live_ws.upstream_dial_failed",
				zap.Int64("account_id", account.ID),
				zap.Int("status", statusCode),
				zap.Error(dialErr))
			if boundAccountID > 0 {
				_ = clientConn.Close(coderws.StatusInternalError, "upstream live dial failed")
				return
			}
			excludedIDs[account.ID] = struct{}{}
			if fs.HandleFailoverError(c.Request.Context(), h.gatewayService, account.ID, account.Platform, account.GetPoolModeRetryCount(), &service.UpstreamFailoverError{StatusCode: max(statusCode, http.StatusBadGateway)}) == FailoverContinue {
				continue
			}
			_ = clientConn.Close(coderws.StatusTryAgainLater, "upstream live dial exhausted")
			return
		}

		setupPayload, _ := service.RewriteGeminiLiveSetupModel(firstFrame, mappedModel)
		writeCtx, cancelWrite := context.WithTimeout(c.Request.Context(), geminiLiveSetupTimeout)
		writeErr := upConn.WriteFrame(writeCtx, firstMsgType, setupPayload)
		cancelWrite()
		if writeErr != nil {
			_ = upConn.CloseNow()
			rel()
			if boundAccountID > 0 {
				_ = clientConn.Close(coderws.StatusInternalError, "failed to send live setup")
				return
			}
			excludedIDs[account.ID] = struct{}{}
			continue
		}

		accountRelease = rel
		upstreamConn = upConn
		break
	}

	defer accountRelease()
	defer func() { _ = upstreamConn.CloseNow() }()

	requestStart := time.Now()
	usageTracker := &service.GeminiLiveUsageTracker{}
	sessionCtx, cancelSession := context.WithCancel(c.Request.Context())
	defer cancelSession()

	var wg sync.WaitGroup
	wg.Add(2)

	// Client -> Upstream
	go func() {
		defer wg.Done()
		defer cancelSession()
		for {
			mt, payload, err := clientConn.Read(sessionCtx)
			if err != nil {
				if !isExpectedGeminiLiveWSClose(err) {
					reqLog.Debug("gemini.live_ws.client_read_end", zap.Error(err))
				}
				return
			}
			if auditBody := service.BuildGeminiLiveAuditJSON(payload); len(auditBody) > 0 {
				if decision := h.checkSecurityAuditStage(c, reqLog, apiKey, subject, "gemini", reqModel, auditBody, "subsequent_turn"); decision != nil && !decision.AllowNextStage {
					msg := decision.ClientMessage
					if msg == "" {
						msg = "message blocked by security policy"
					}
					_ = clientConn.Close(coderws.StatusPolicyViolation, msg)
					return
				}
			}
			if writeErr := upstreamConn.WriteFrame(sessionCtx, mt, payload); writeErr != nil {
				return
			}
		}
	}()

	// Upstream -> Client + Live UsageMeta Extraction
	go func() {
		defer wg.Done()
		defer cancelSession()
		for {
			mt, payload, err := upstreamConn.ReadFrame(sessionCtx)
			if err != nil {
				if !isExpectedGeminiLiveWSClose(err) {
					reqLog.Debug("gemini.live_ws.upstream_read_end", zap.Error(err))
				}
				return
			}
			usageTracker.ObserveServerFrame(payload)
			if writeErr := clientConn.Write(sessionCtx, mt, payload); writeErr != nil {
				return
			}
		}
	}()

	wg.Wait()

	totalUsage := usageTracker.Total()
	if hasBillableGeminiUsage(totalUsage) {
		sessionReqID := "live-" + uuid.NewString()
		sum := sha256.Sum256([]byte(sessionReqID))
		payloadHash := hex.EncodeToString(sum[:16])
		result := &service.ForwardResult{
			RequestID:     sessionReqID,
			Usage:         totalUsage,
			Model:         reqModel,
			UpstreamModel: mappedModel,
			Duration:      time.Since(requestStart),
			Stream:        true,
		}
		channelFields := channelMapping.ToUsageFields(reqModel, mappedModel)
		upstreamEndpoint := "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
		if constrained && rawEphemeralToken != "" {
			upstreamEndpoint = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained"
		}
		billingSub := h.resolveGeminiExtSubscription(c, apiKey)
		inboundEndpoint := c.Request.URL.Path
		userAgent := c.GetHeader("User-Agent")
		h.submitUsageRecordTask(context.WithoutCancel(c.Request.Context()), func(ctx context.Context) {
			recCtx, cancel := context.WithTimeout(ctx, geminiExtOpTimeout)
			defer cancel()
			if err := h.gatewayService.RecordUsage(recCtx, &service.RecordUsageInput{
				Result:             result,
				APIKey:             apiKey,
				User:               apiKey.User,
				Account:            account,
				Subscription:       billingSub,
				InboundEndpoint:    inboundEndpoint,
				UpstreamEndpoint:   upstreamEndpoint,
				ChannelUsageFields: channelFields,
				RequestPayloadHash: payloadHash,
				PricingAt:          requestStart,
				APIKeyService:      h.apiKeyService,
				UserAgent:          userAgent,
			}); err != nil {
				logger.LegacyPrintf("handler.gemini_live_ws", "record gemini live usage failed: %v", err)
			}
		})
	}
}
