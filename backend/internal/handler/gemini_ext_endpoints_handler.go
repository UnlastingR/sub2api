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
	"time"

	pkghttputil "github.com/Wei-Shaw/sub2api/internal/pkg/httputil"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	middleware2 "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"go.uber.org/zap"
)

const geminiExtOpTimeout = 10 * time.Second

// geminiExtRequestOpts configures shared auth/audit/pinning/failover handling for Gemini AI Studio extensions.
type geminiExtRequestOpts struct {
	Component         string
	Method            string
	UpstreamPath      string
	UpstreamPathFn    func(mappedModel string) string
	ReadBody          bool
	RequireAudit      bool
	RequireBilling    bool
	ExplicitModel     string
	BindKind          string
	PinKind           string
	PinID             string
	ListFields        []string
	UseUserSticky     bool
	ChannelModelCheck bool
	IsStream          bool
	IsDownload        bool
	UploadSession     *service.GeminiUploadSession
	TransformJSON     func(ctx context.Context, apiKey *service.APIKey, subject middleware2.AuthSubject, account *service.Account, reqModel string, respBody []byte) ([]byte, error)
	OnJSONSuccess     func(ctx context.Context, apiKey *service.APIKey, subject middleware2.AuthSubject, subscription *service.UserSubscription, account *service.Account, reqModel string, body []byte, respBody []byte, result *service.ForwardResult)
}

// ResolveGeminiAuthTokenMeta looks up issuing gateway metadata for a Live Constrained ephemeral token.
func (h *GatewayHandler) ResolveGeminiAuthTokenMeta(ctx context.Context, rawToken string) (*service.GeminiAuthTokenMeta, bool) {
	if h == nil || h.geminiCompatService == nil {
		return nil, false
	}
	return h.geminiCompatService.GetGeminiAuthTokenMeta(ctx, rawToken)
}

func (h *GatewayHandler) checkGeminiGroupModelAllowlist(c *gin.Context, apiKey *service.APIKey, model string) bool {
	if model == "" || apiKey == nil || apiKey.Group == nil || !apiKey.Group.ModelAllowlistEnabled() {
		return true
	}
	if apiKey.Group.ModelAllowlist.Allows(model) {
		return true
	}
	googleError(c, http.StatusForbidden, fmt.Sprintf("Model '%s' is not allowed by group model allowlist", model))
	return false
}

func (h *GatewayHandler) resolveGeminiExtSubscription(c *gin.Context, apiKey *service.APIKey) *service.UserSubscription {
	if apiKey == nil {
		return nil
	}
	subscription, _ := middleware2.GetSubscriptionFromContext(c)
	return subscription
}

func (h *GatewayHandler) resolvePinnedGeminiExtAccount(
	c *gin.Context,
	reqLog *zap.Logger,
	kind, id string,
	callerUserID int64,
	requireFound bool,
) (accountID int64, found bool, blocked bool) {
	if kind == "" || id == "" || h.geminiCompatService == nil {
		return 0, false, false
	}
	accID, boundUserID, ok := h.geminiCompatService.ResolveGeminiExtResource(c.Request.Context(), kind, id)
	if !ok {
		if requireFound && h.geminiCompatService.HasGeminiExtCache() {
			googleError(c, http.StatusNotFound, "Requested resource was not found")
			return 0, false, true
		}
		return 0, false, false
	}
	if boundUserID > 0 && boundUserID != callerUserID {
		reqLog.Warn("gemini.ext.cross_user_resource_denied",
			zap.String("kind", kind),
			zap.Int64("owner_user_id", boundUserID),
			zap.Int64("caller_user_id", callerUserID))
		googleError(c, http.StatusNotFound, "Requested resource was not found")
		return 0, true, true
	}
	return accID, true, false
}

func (h *GatewayHandler) loadValidatedPinnedGeminiAccount(ctx context.Context, apiKey *service.APIKey, accountID int64) (*service.Account, bool) {
	if h == nil || h.geminiCompatService == nil || accountID <= 0 {
		return nil, false
	}
	pinnedAcc, err := h.geminiCompatService.GetAccountByID(ctx, accountID)
	if err != nil || pinnedAcc == nil || !service.IsGeminiAIStudioEndpointAccount(pinnedAcc) || !pinnedAcc.IsSchedulable() {
		return nil, false
	}
	if apiKey != nil && apiKey.GroupID != nil && *apiKey.GroupID > 0 && !pinnedAcc.BelongsToGroup(*apiKey.GroupID) {
		return nil, false
	}
	return pinnedAcc, true
}

func (h *GatewayHandler) executeGeminiExtRequest(c *gin.Context, opts geminiExtRequestOpts) {
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
		opts.Component,
		zap.Int64("user_id", subject.UserID),
		zap.Int64("api_key_id", apiKey.ID),
		zap.Any("group_id", apiKey.GroupID),
	)
	if h.geminiCompatService == nil || h.gatewayService == nil {
		googleError(c, http.StatusServiceUnavailable, "Gemini compatibility service unavailable")
		return
	}
	// Resource ownership cannot be verified without the binding cache.
	if !h.geminiCompatService.HasGeminiExtCache() {
		googleError(c, http.StatusServiceUnavailable, "Gemini resource ownership cache unavailable")
		return
	}

	var body []byte
	if opts.ReadBody && c.Request.Body != nil {
		var err error
		body, err = pkghttputil.ReadRequestBodyWithPrealloc(c.Request)
		if err != nil {
			var maxErr *http.MaxBytesError
			if errors.As(err, &maxErr) {
				googleError(c, http.StatusRequestEntityTooLarge, buildBodyTooLargeMessage(maxErr.Limit))
				return
			}
			googleError(c, http.StatusBadRequest, "Failed to read request body")
			return
		}
	}

	reqModel := strings.TrimSpace(opts.ExplicitModel)
	if reqModel == "" && len(body) > 0 {
		reqModel = service.ExtractGeminiAIStudioBodyModel(c.Request.URL.Path, body)
	}
	if !h.checkGeminiGroupModelAllowlist(c, apiKey, reqModel) {
		return
	}
	if opts.RequireAudit {
		if decision := h.checkSecurityAudit(c, reqLog, apiKey, subject, "gemini", reqModel, body); decision != nil && !decision.AllowNextStage {
			googleSecurityAuditError(c, decision)
			return
		}
	}

	subscription := h.resolveGeminiExtSubscription(c, apiKey)
	if opts.RequireBilling && h.billingCacheService != nil && apiKey.User != nil {
		quotaPlatform := service.QuotaPlatform(c.Request.Context(), apiKey)
		if err := h.billingCacheService.CheckBillingEligibility(c.Request.Context(), apiKey.User, apiKey, apiKey.Group, subscription, quotaPlatform); err != nil {
			status, _, message, _ := billingErrorDetails(err)
			googleError(c, status, message)
			return
		}
	}

	var boundAccountID int64
	if opts.UploadSession != nil {
		boundAccountID = opts.UploadSession.AccountID
	} else if opts.PinKind != "" && opts.PinID != "" {
		accID, found, blocked := h.resolvePinnedGeminiExtAccount(c, reqLog, opts.PinKind, opts.PinID, subject.UserID, true)
		if blocked {
			return
		}
		if found {
			boundAccountID = accID
		}
	}

	// 若请求体引用了已上传文件（如 files/... 批量输入或 RAG 导入），强校验归属并优先绑定到同账号。
	if len(body) > 0 {
		if refKind, refID := service.ExtractGeminiExtReferencedResource(body); refKind != "" && refID != "" {
			refAccID, found, blocked := h.resolvePinnedGeminiExtAccount(c, reqLog, refKind, refID, subject.UserID, true)
			if blocked {
				return
			}
			if found {
				if boundAccountID > 0 && boundAccountID != refAccID {
					googleError(c, http.StatusConflict, "Referenced resource belongs to a different upstream account")
					return
				}
				boundAccountID = refAccID
			}
		}
	}

	// 用户并发槽位
	userRelease, err := h.concurrencyHelper.AcquireUserSlotWithWait(c, subject.UserID, subject.Concurrency, false, nil)
	if err != nil {
		h.handleConcurrencyError(c, err, "user", false)
		return
	}
	defer userRelease()

	if reqModel != "" {
		ensureCompositeTargetPlatform(c, apiKey, reqModel)
		if !compositeTargetPlatformAllowed(c, apiKey, reqModel, service.PlatformGemini) {
			googleError(c, http.StatusNotFound, "Model is not routed to Gemini platform in this composite group")
			return
		}
		if upstreamModel, ok := service.ResolvedUpstreamModelFromContext(c.Request.Context()); ok && upstreamModel != "" {
			reqModel = upstreamModel
		}
		channelMapping, _ := h.gatewayService.ResolveChannelMappingAndRestrict(c.Request.Context(), apiKey.GroupID, reqModel)
		if channelMapping.Mapped && channelMapping.MappedModel != "" {
			reqModel = channelMapping.MappedModel
		}
	}

	sessionHash := ""
	if boundAccountID == 0 && opts.UseUserSticky {
		sessionHash = "gemini:ext:user:" + strconv.FormatInt(subject.UserID, 10)
	}

	geminiConcurrency := NewConcurrencyHelper(h.concurrencyHelper.concurrencyService, SSEPingFormatNone, 0)
	excludedIDs, exclErr := h.gatewayService.GeminiAIStudioEndpointExclusions(c.Request.Context(), apiKey.GroupID, boundAccountID)
	if exclErr != nil {
		reqLog.Warn("gemini.ext.exclusions_failed", zap.Error(exclErr))
	}
	if excludedIDs == nil {
		excludedIDs = make(map[int64]struct{})
	}

	maxSwitches := h.maxAccountSwitchesGemini
	if boundAccountID > 0 {
		maxSwitches = 0
	}
	fs := NewFailoverState(maxSwitches, boundAccountID > 0)
	modelForSelection := reqModel
	if modelForSelection == "" {
		modelForSelection = "gemini-3.5-flash"
	}

	var account *service.Account
	streamStarted := false
	for {
		var selection *service.AccountSelectionResult
		var selErr error
		if boundAccountID > 0 {
			pinnedAcc, ok := h.loadValidatedPinnedGeminiAccount(c.Request.Context(), apiKey, boundAccountID)
			if !ok {
				googleError(c, http.StatusServiceUnavailable, "Pinned Gemini AI Studio account is unavailable")
				return
			}
			account = pinnedAcc
		} else {
			selection, selErr = h.gatewayService.SelectAccountWithLoadAwareness(
				c.Request.Context(), apiKey.GroupID, sessionHash, modelForSelection, excludedIDs, "", subject.UserID,
			)
			if selErr != nil || selection == nil || selection.Account == nil {
				switch fs.HandleSelectionExhausted(c.Request.Context()) {
				case FailoverContinue:
					continue
				case FailoverCanceled:
					return
				default:
					if fs.LastFailoverErr != nil {
						h.handleGeminiFailoverExhausted(c, fs.LastFailoverErr)
					} else {
						markOpsRoutingCapacityLimited(c)
						googleError(c, http.StatusServiceUnavailable, "No available Gemini AI Studio accounts")
					}
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
				googleError(c, http.StatusServiceUnavailable, "No available Gemini AI Studio accounts")
				return
			}
			continue
		}
		account = latest

		var accountRelease func()
		if selection != nil {
			rel, acquired := h.acquireGeminiEndpointAccountSlot(c, reqLog, geminiConcurrency, selection, false, &streamStarted)
			if !acquired {
				return
			}
			accountRelease = rel
		} else {
			rel, acquired, slotErr := geminiConcurrency.TryAcquireAccountSlot(c.Request.Context(), account.ID, account.Concurrency)
			if slotErr != nil {
				h.handleConcurrencyError(c, slotErr, "account", false)
				return
			}
			if !acquired {
				googleError(c, http.StatusTooManyRequests, "Pinned upstream account concurrency limit reached")
				return
			}
			accountRelease = rel
		}

		mappedModel := reqModel
		if reqModel != "" {
			mappedModel = account.GetMappedModel(reqModel)
		}
		upstreamPath := opts.UpstreamPath
		if opts.UpstreamPathFn != nil {
			upstreamPath = opts.UpstreamPathFn(mappedModel)
		}

		var forwardErr error
		if opts.IsStream {
			forwardErr = h.geminiCompatService.ForwardGeminiAIStudioStream(c.Request.Context(), c, account, service.GeminiAIStudioStreamSpec{
				Method:        opts.Method,
				Path:          upstreamPath,
				Query:         c.Request.URL.Query(),
				UserID:        subject.UserID,
				BindKind:      opts.BindKind,
				IsDownload:    opts.IsDownload,
				UploadSession: opts.UploadSession,
				OriginalModel: reqModel,
				MappedModel:   mappedModel,
			})
			accountRelease()
			if forwardErr == nil {
				return
			}
		} else {
			var transformFn func([]byte) ([]byte, error)
			if opts.TransformJSON != nil || (opts.Method == http.MethodGet && len(opts.ListFields) > 0) {
				listFields := opts.ListFields
				transformFn = func(respBody []byte) ([]byte, error) {
					out := respBody
					var tfErr error
					if opts.Method == http.MethodGet && len(listFields) > 0 {
						out, tfErr = h.geminiCompatService.FilterGeminiExtListByUser(c.Request.Context(), out, listFields, opts.BindKind, subject.UserID)
						if tfErr != nil {
							return nil, tfErr
						}
					}
					if opts.TransformJSON != nil {
						out, tfErr = opts.TransformJSON(c.Request.Context(), apiKey, subject, account, reqModel, out)
						if tfErr != nil {
							return nil, tfErr
						}
					}
					return out, nil
				}
			}
			result, respBody, err := h.geminiCompatService.ForwardGeminiAIStudioExtensionJSON(
				c.Request.Context(),
				c,
				account,
				service.GeminiAIStudioExtJSONSpec{
					Method:                opts.Method,
					Path:                  upstreamPath,
					Query:                 c.Request.URL.Query(),
					Body:                  body,
					OriginalModel:         reqModel,
					MappedModel:           mappedModel,
					UserID:                subject.UserID,
					BindKind:              opts.BindKind,
					TransformResponseJSON: transformFn,
				},
			)
			accountRelease()
			if err == nil {
				if opts.OnJSONSuccess != nil {
					opts.OnJSONSuccess(c.Request.Context(), apiKey, subject, subscription, account, reqModel, body, respBody, result)
				}
				return
			}
			forwardErr = err
		}

		var failoverErr *service.UpstreamFailoverError
		if errors.As(forwardErr, &failoverErr) && boundAccountID == 0 {
			switch fs.HandleFailoverError(c.Request.Context(), h.gatewayService, account.ID, account.Platform, account.GetPoolModeRetryCount(), failoverErr) {
			case FailoverContinue:
				excludedIDs[account.ID] = struct{}{}
				continue
			case FailoverExhausted:
				h.handleGeminiFailoverExhausted(c, fs.LastFailoverErr)
				return
			case FailoverCanceled:
				return
			}
		}
		if failoverErr != nil {
			h.handleGeminiFailoverExhausted(c, failoverErr)
			return
		}
		return
	}
}

// --- Managed Agents: Agents, Environments, Credentials, Triggers, Webhooks ---

func (h *GatewayHandler) GeminiV1BetaAgents(c *gin.Context) {
	isPost := c.Request.Method == http.MethodPost
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:     "handler.gemini_v1beta.agents",
		Method:        c.Request.Method,
		UpstreamPath:  "/v1beta/agents",
		ReadBody:      isPost,
		RequireAudit:  isPost,
		BindKind:      "agent",
		ListFields:    []string{"agents"},
		UseUserSticky: true,
	})
}

func (h *GatewayHandler) GeminiV1BetaAgentByID(c *gin.Context) {
	agentID := strings.TrimSpace(c.Param("agentID"))
	if !service.IsSafeGeminiExtID(agentID) {
		googleError(c, http.StatusBadRequest, "Invalid agent ID")
		return
	}
	isPatch := c.Request.Method == http.MethodPatch
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:    "handler.gemini_v1beta.agent",
		Method:       c.Request.Method,
		UpstreamPath: "/v1beta/agents/" + agentID,
		ReadBody:     isPatch,
		RequireAudit: isPatch,
		BindKind:     "agent",
		PinKind:      "agent",
		PinID:        agentID,
	})
}

func (h *GatewayHandler) GeminiV1BetaEnvironments(c *gin.Context) {
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:     "handler.gemini_v1beta.environments",
		Method:        c.Request.Method,
		UpstreamPath:  "/v1beta/environments",
		ReadBody:      c.Request.Method == http.MethodPost,
		BindKind:      "env",
		ListFields:    []string{"environments"},
		UseUserSticky: true,
	})
}

func (h *GatewayHandler) GeminiV1BetaEnvironmentAction(c *gin.Context) {
	rawAction := strings.TrimPrefix(strings.TrimSpace(c.Param("envAction")), "/")
	if rawAction == "" {
		googleError(c, http.StatusBadRequest, "Environment path is required")
		return
	}
	var envID string
	var isFileTransfer bool
	if id, _, ok := service.ParseGeminiExtIDAction(rawAction, "get", "delete"); ok {
		envID = id
	} else if left, sub, hasSlash := strings.Cut(rawAction, "/"); hasSlash && service.IsSafeGeminiExtID(left) {
		envID = left
		if sub == "files" {
			isFileTransfer = false
		} else if subpath, ok := strings.CutPrefix(sub, "files/"); ok && service.IsSafeGeminiExtSubpath(subpath) {
			isFileTransfer = true
		} else {
			googleError(c, http.StatusBadRequest, "Invalid environment file path")
			return
		}
	} else {
		googleError(c, http.StatusBadRequest, "Invalid environment action path")
		return
	}

	if isFileTransfer {
		h.executeGeminiExtRequest(c, geminiExtRequestOpts{
			Component:    "handler.gemini_v1beta.environment.files",
			Method:       c.Request.Method,
			UpstreamPath: "/v1beta/environments/" + rawAction,
			BindKind:     "env",
			PinKind:      "env",
			PinID:        envID,
			IsStream:     true,
			IsDownload:   c.Request.Method == http.MethodGet,
		})
		return
	}

	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:    "handler.gemini_v1beta.environment",
		Method:       c.Request.Method,
		UpstreamPath: "/v1beta/environments/" + rawAction,
		ReadBody:     c.Request.Method == http.MethodPost || c.Request.Method == http.MethodPatch || c.Request.Method == http.MethodPut,
		BindKind:     "env",
		PinKind:      "env",
		PinID:        envID,
	})
}

func (h *GatewayHandler) GeminiUploadV1BetaEnvironments(c *gin.Context) {
	if sess := h.tryResolveGeminiUploadSession(c); sess != nil {
		h.executeGeminiExtRequest(c, geminiExtRequestOpts{
			Component:     "handler.gemini_upload.environments.resumable",
			Method:        c.Request.Method,
			UpstreamPath:  "/upload/v1beta/environments/" + strings.TrimPrefix(c.Param("envAction"), "/"),
			IsStream:      true,
			UploadSession: sess,
		})
		return
	} else if c.Query("upload_id") != "" {
		return
	}

	rawAction := strings.TrimPrefix(strings.TrimSpace(c.Param("envAction")), "/")
	envID, sub, hasSlash := strings.Cut(rawAction, "/")
	if !hasSlash || !service.IsSafeGeminiExtID(envID) {
		googleError(c, http.StatusBadRequest, "Invalid environment upload path")
		return
	}
	if sub != "files" {
		if subpath, ok := strings.CutPrefix(sub, "files/"); !ok || !service.IsSafeGeminiExtSubpath(subpath) {
			googleError(c, http.StatusBadRequest, "Invalid environment upload file path")
			return
		}
	}
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:     "handler.gemini_upload.environments",
		Method:        c.Request.Method,
		UpstreamPath:  "/upload/v1beta/environments/" + rawAction,
		BindKind:      "env",
		PinKind:       "env",
		PinID:         envID,
		UseUserSticky: true,
		IsStream:      true,
	})
}

func (h *GatewayHandler) GeminiV1BetaCredentials(c *gin.Context) {
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:     "handler.gemini_v1beta.credentials",
		Method:        c.Request.Method,
		UpstreamPath:  "/v1beta/credentials",
		ReadBody:      c.Request.Method == http.MethodPost,
		BindKind:      "cred",
		ListFields:    []string{"credentials"},
		UseUserSticky: true,
	})
}

func (h *GatewayHandler) GeminiV1BetaCredentialByID(c *gin.Context) {
	credID := strings.TrimSpace(c.Param("credentialID"))
	if !service.IsSafeGeminiExtID(credID) {
		googleError(c, http.StatusBadRequest, "Invalid credential ID")
		return
	}
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:    "handler.gemini_v1beta.credential",
		Method:       c.Request.Method,
		UpstreamPath: "/v1beta/credentials/" + credID,
		ReadBody:     c.Request.Method == http.MethodPatch,
		BindKind:     "cred",
		PinKind:      "cred",
		PinID:        credID,
	})
}

func (h *GatewayHandler) GeminiV1BetaTriggers(c *gin.Context) {
	isPost := c.Request.Method == http.MethodPost
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:     "handler.gemini_v1beta.triggers",
		Method:        c.Request.Method,
		UpstreamPath:  "/v1beta/triggers",
		ReadBody:      isPost,
		RequireAudit:  isPost,
		BindKind:      "trigger",
		ListFields:    []string{"triggers"},
		UseUserSticky: true,
	})
}

func (h *GatewayHandler) GeminiV1BetaTriggerAction(c *gin.Context) {
	rawAction := strings.TrimPrefix(strings.TrimSpace(c.Param("triggerAction")), "/")
	if rawAction == "" {
		googleError(c, http.StatusBadRequest, "Trigger ID is required")
		return
	}
	triggerID := rawAction
	if left, sub, hasSlash := strings.Cut(rawAction, "/"); hasSlash {
		triggerID = left
		if !service.IsSafeGeminiExtID(triggerID) {
			googleError(c, http.StatusBadRequest, "Invalid trigger ID")
			return
		}
		if sub != "executions" {
			execID, ok := strings.CutPrefix(sub, "executions/")
			if !ok || !service.IsSafeGeminiExtID(execID) {
				googleError(c, http.StatusBadRequest, "Invalid trigger execution path")
				return
			}
		}
	} else if !service.IsSafeGeminiExtID(triggerID) {
		googleError(c, http.StatusBadRequest, "Invalid trigger ID")
		return
	}
	hasBody := c.Request.Method == http.MethodPost || c.Request.Method == http.MethodPatch
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:    "handler.gemini_v1beta.trigger",
		Method:       c.Request.Method,
		UpstreamPath: "/v1beta/triggers/" + rawAction,
		ReadBody:     hasBody,
		RequireAudit: hasBody,
		BindKind:     "trigger",
		PinKind:      "trigger",
		PinID:        triggerID,
	})
}

func (h *GatewayHandler) GeminiV1BetaWebhooks(c *gin.Context) {
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:     "handler.gemini_v1beta.webhooks",
		Method:        c.Request.Method,
		UpstreamPath:  "/v1beta/webhooks",
		ReadBody:      c.Request.Method == http.MethodPost,
		BindKind:      "webhook",
		ListFields:    []string{"webhooks"},
		UseUserSticky: true,
	})
}

func (h *GatewayHandler) GeminiV1BetaWebhookByID(c *gin.Context) {
	rawParam := strings.TrimSpace(c.Param("webhookID"))
	webhookID, action, ok := service.ParseGeminiExtIDAction(rawParam, "ping", "rotateSigningSecret")
	if !ok {
		googleError(c, http.StatusBadRequest, "Invalid webhook action")
		return
	}
	if action != "" && c.Request.Method != http.MethodPost {
		googleError(c, http.StatusMethodNotAllowed, "Webhook action requires POST")
		return
	}
	if action == "" && c.Request.Method == http.MethodPost {
		googleError(c, http.StatusMethodNotAllowed, "POST requires :ping or :rotateSigningSecret action")
		return
	}
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:    "handler.gemini_v1beta.webhook",
		Method:       c.Request.Method,
		UpstreamPath: "/v1beta/webhooks/" + rawParam,
		ReadBody:     c.Request.Method == http.MethodPatch || c.Request.Method == http.MethodPost,
		BindKind:     "webhook",
		PinKind:      "webhook",
		PinID:        webhookID,
	})
}

// --- Live API Ephemeral Auth Tokens ---

func (h *GatewayHandler) GeminiV1BetaAuthTokensCreate(c *gin.Context) {
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:      "handler.gemini_v1beta.auth_tokens",
		Method:         http.MethodPost,
		UpstreamPath:   "/v1beta/auth_tokens",
		ReadBody:       true,
		RequireBilling: true,
		UseUserSticky:  true,
		TransformJSON: func(_ context.Context, apiKey *service.APIKey, subject middleware2.AuthSubject, account *service.Account, reqModel string, respBody []byte) ([]byte, error) {
			upstreamToken := strings.TrimSpace(gjson.GetBytes(respBody, "name").String())
			if upstreamToken == "" || h.geminiCompatService == nil {
				return respBody, nil
			}
			var expUnix int64
			if expStr := strings.TrimSpace(gjson.GetBytes(respBody, "expireTime").String()); expStr != "" {
				if parsed, err := time.Parse(time.RFC3339, expStr); err == nil {
					expUnix = parsed.Unix()
				}
			}
			var groupID int64
			if apiKey != nil && apiKey.GroupID != nil {
				groupID = *apiKey.GroupID
			}
			gatewayToken := "auth_tokens/gw-" + strings.ReplaceAll(uuid.NewString(), "-", "")
			saveCtx, cancel := context.WithTimeout(context.Background(), geminiExtOpTimeout)
			defer cancel()
			if saveErr := h.geminiCompatService.SaveGeminiAuthTokenMeta(saveCtx, gatewayToken, service.GeminiAuthTokenMeta{
				AccountID:     account.ID,
				UserID:        subject.UserID,
				APIKeyID:      apiKey.ID,
				GroupID:       groupID,
				Model:         reqModel,
				UpstreamToken: upstreamToken,
				ExpiresAt:     expUnix,
			}); saveErr != nil {
				return nil, saveErr
			}
			return sjson.SetBytes(respBody, "name", gatewayToken)
		},
	})
}

// --- File API & File Search Stores ---

func (h *GatewayHandler) tryResolveGeminiUploadSession(c *gin.Context) *service.GeminiUploadSession {
	uploadID := strings.TrimSpace(c.Query("upload_id"))
	if uploadID == "" {
		return nil
	}
	if !service.IsSafeGeminiExtID(uploadID) || h.geminiCompatService == nil {
		googleError(c, http.StatusBadRequest, "Invalid upload_id")
		return nil
	}
	subject, ok := middleware2.GetAuthSubjectFromContext(c)
	if !ok {
		googleError(c, http.StatusInternalServerError, "User context not found")
		return nil
	}
	sess, found := h.geminiCompatService.GetGeminiUploadSession(c.Request.Context(), uploadID)
	if !found || (sess.UserID > 0 && sess.UserID != subject.UserID) {
		googleError(c, http.StatusNotFound, "Upload session not found or expired")
		return nil
	}
	return sess
}

func (h *GatewayHandler) GeminiV1BetaFilesList(c *gin.Context) {
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:     "handler.gemini_v1beta.files.list",
		Method:        http.MethodGet,
		UpstreamPath:  "/v1beta/files",
		BindKind:      "file",
		ListFields:    []string{"files"},
		UseUserSticky: true,
	})
}

func (h *GatewayHandler) GeminiV1BetaFilesCreate(c *gin.Context) {
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:      "handler.gemini_v1beta.files.create",
		Method:         http.MethodPost,
		UpstreamPath:   "/v1beta/files",
		ReadBody:       true,
		RequireBilling: true,
		BindKind:       "file",
		UseUserSticky:  true,
	})
}

func (h *GatewayHandler) GeminiV1BetaFilesRegister(c *gin.Context) {
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:     "handler.gemini_v1beta.files.register",
		Method:        http.MethodPost,
		UpstreamPath:  "/v1beta/files:register",
		ReadBody:      true,
		BindKind:      "file",
		UseUserSticky: true,
	})
}

func (h *GatewayHandler) GeminiV1BetaFileAction(c *gin.Context) {
	rawAction := strings.TrimPrefix(strings.TrimSpace(c.Param("fileAction")), "/")
	fileID, action, ok := service.ParseGeminiExtIDAction(rawAction, "download")
	if !ok {
		googleError(c, http.StatusBadRequest, "Invalid file path")
		return
	}
	if action == "download" {
		if c.Request.Method != http.MethodGet {
			googleError(c, http.StatusMethodNotAllowed, "File download requires GET")
			return
		}
		h.executeGeminiExtRequest(c, geminiExtRequestOpts{
			Component:    "handler.gemini_v1beta.files.download",
			Method:       http.MethodGet,
			UpstreamPath: "/v1beta/files/" + rawAction,
			BindKind:     "file",
			PinKind:      "file",
			PinID:        fileID,
			IsStream:     true,
			IsDownload:   true,
		})
		return
	}
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:    "handler.gemini_v1beta.file",
		Method:       c.Request.Method,
		UpstreamPath: "/v1beta/files/" + rawAction,
		ReadBody:     c.Request.Method == http.MethodPost,
		BindKind:     "file",
		PinKind:      "file",
		PinID:        fileID,
	})
}

func (h *GatewayHandler) GeminiUploadV1BetaFiles(c *gin.Context) {
	rawAction := strings.TrimPrefix(strings.TrimSpace(c.Param("fileAction")), "/")
	upstreamPath := "/upload/v1beta/files"
	var pinID string
	if rawAction != "" {
		if !service.IsSafeGeminiExtID(rawAction) {
			googleError(c, http.StatusBadRequest, "Invalid upload file ID")
			return
		}
		upstreamPath += "/" + rawAction
		pinID = rawAction
	}
	if sess := h.tryResolveGeminiUploadSession(c); sess != nil {
		h.executeGeminiExtRequest(c, geminiExtRequestOpts{
			Component:     "handler.gemini_upload.files.resumable",
			Method:        c.Request.Method,
			UpstreamPath:  upstreamPath,
			BindKind:      "file",
			IsStream:      true,
			UploadSession: sess,
		})
		return
	} else if c.Query("upload_id") != "" {
		return
	}

	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:      "handler.gemini_upload.files",
		Method:         c.Request.Method,
		UpstreamPath:   upstreamPath,
		RequireBilling: true,
		BindKind:       "file",
		PinKind:        "file",
		PinID:          pinID,
		UseUserSticky:  true,
		IsStream:       true,
	})
}

func (h *GatewayHandler) GeminiDownloadV1BetaFiles(c *gin.Context) {
	rawAction := strings.TrimPrefix(strings.TrimSpace(c.Param("fileAction")), "/")
	fileID, _, ok := service.ParseGeminiExtIDAction(rawAction, "download")
	if !ok {
		googleError(c, http.StatusBadRequest, "Invalid download file path")
		return
	}
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:    "handler.gemini_download.files",
		Method:       http.MethodGet,
		UpstreamPath: "/download/v1beta/files/" + rawAction,
		BindKind:     "file",
		PinKind:      "file",
		PinID:        fileID,
		IsStream:     true,
		IsDownload:   true,
	})
}

func (h *GatewayHandler) GeminiV1BetaFileSearchStores(c *gin.Context) {
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:     "handler.gemini_v1beta.file_search_stores",
		Method:        c.Request.Method,
		UpstreamPath:  "/v1beta/fileSearchStores",
		ReadBody:      c.Request.Method == http.MethodPost,
		BindKind:      "store",
		ListFields:    []string{"fileSearchStores"},
		UseUserSticky: true,
	})
}

func (h *GatewayHandler) GeminiV1BetaFileSearchStoreAction(c *gin.Context) {
	rawAction := strings.TrimPrefix(strings.TrimSpace(c.Param("storeAction")), "/")
	if rawAction == "" {
		googleError(c, http.StatusBadRequest, "FileSearchStore path is required")
		return
	}
	var storeID, colonAction string
	if id, action, ok := service.ParseGeminiExtIDAction(rawAction, "importFile", "uploadToFileSearchStore"); ok {
		storeID = id
		colonAction = action
	} else if left, sub, hasSlash := strings.Cut(rawAction, "/"); hasSlash && service.IsSafeGeminiExtID(left) {
		storeID = left
		switch {
		case sub == "documents":
		case strings.HasPrefix(sub, "documents/") && service.IsSafeGeminiExtID(strings.TrimPrefix(sub, "documents/")):
		case strings.HasPrefix(sub, "operations/") && service.IsSafeGeminiExtID(strings.TrimPrefix(sub, "operations/")):
		case strings.HasPrefix(sub, "upload/operations/") && service.IsSafeGeminiExtID(strings.TrimPrefix(sub, "upload/operations/")):
		default:
			googleError(c, http.StatusBadRequest, "Invalid fileSearchStore subpath")
			return
		}
	} else {
		googleError(c, http.StatusBadRequest, "Invalid fileSearchStore path")
		return
	}

	if colonAction == "uploadToFileSearchStore" {
		h.executeGeminiExtRequest(c, geminiExtRequestOpts{
			Component:      "handler.gemini_v1beta.file_search_store.upload",
			Method:         c.Request.Method,
			UpstreamPath:   "/v1beta/fileSearchStores/" + rawAction,
			RequireBilling: true,
			BindKind:       "store",
			PinKind:        "store",
			PinID:          storeID,
			IsStream:       true,
		})
		return
	}

	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:    "handler.gemini_v1beta.file_search_store",
		Method:       c.Request.Method,
		UpstreamPath: "/v1beta/fileSearchStores/" + rawAction,
		ReadBody:     c.Request.Method == http.MethodPost || c.Request.Method == http.MethodPatch,
		BindKind:     "store",
		PinKind:      "store",
		PinID:        storeID,
	})
}

func (h *GatewayHandler) GeminiUploadV1BetaFileSearchStores(c *gin.Context) {
	rawAction := strings.TrimPrefix(strings.TrimSpace(c.Param("storeAction")), "/")
	storeID, action, ok := service.ParseGeminiExtIDAction(rawAction, "uploadToFileSearchStore")
	if !ok || action != "uploadToFileSearchStore" {
		googleError(c, http.StatusBadRequest, "Invalid fileSearchStore upload path")
		return
	}
	if sess := h.tryResolveGeminiUploadSession(c); sess != nil {
		h.executeGeminiExtRequest(c, geminiExtRequestOpts{
			Component:     "handler.gemini_upload.file_search_stores.resumable",
			Method:        c.Request.Method,
			UpstreamPath:  "/upload/v1beta/fileSearchStores/" + rawAction,
			BindKind:      "store",
			IsStream:      true,
			UploadSession: sess,
		})
		return
	} else if c.Query("upload_id") != "" {
		return
	}

	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:      "handler.gemini_upload.file_search_stores",
		Method:         c.Request.Method,
		UpstreamPath:   "/upload/v1beta/fileSearchStores/" + rawAction,
		RequireBilling: true,
		BindKind:       "store",
		PinKind:        "store",
		PinID:          storeID,
		UseUserSticky:  true,
		IsStream:       true,
	})
}

// --- Batch API (50% Asynchronous Settlement) ---

func isGeminiBatchModelAction(action string) bool {
	return action == "batchGenerateContent" || action == "asyncBatchEmbedContent" || action == "asyncBatchEmbedContents"
}

func buildGeminiBatchUsageSnapshotFromMeta(meta *service.GeminiBatchMeta) geminiInteractionUsageSnapshot {
	pricingAt := time.Now()
	if meta.CreatedAtUnix > 0 {
		pricingAt = time.Unix(meta.CreatedAtUnix, 0)
	}
	var groupID *int64
	if meta.GroupID > 0 {
		gid := meta.GroupID
		groupID = &gid
	}
	var payloadHash string
	if meta.BatchKey != "" {
		sum := sha256.Sum256([]byte("gemini-batch:" + meta.BatchKey))
		payloadHash = hex.EncodeToString(sum[:16])
	}
	return geminiInteractionUsageSnapshot{
		parentCtx:          context.Background(),
		apiKey:             &service.APIKey{ID: meta.APIKeyID, UserID: meta.UserID, GroupID: groupID},
		account:            &service.Account{ID: meta.AccountID, Platform: service.PlatformGemini},
		userID:             meta.UserID,
		pricingAt:          pricingAt,
		requestModel:       meta.OriginalModel,
		upstreamModel:      meta.MappedModel,
		channelUsageFields: meta.ChannelUsageFields,
		requestPayloadHash: payloadHash,
		inboundEndpoint:    meta.InboundEndpoint,
		upstreamEndpoint:   meta.UpstreamEndpoint,
		quotaPlatform:      service.PlatformGemini,
		userAgent:          meta.UserAgent,
	}
}

func (h *GatewayHandler) runMandatoryGeminiBatchRecordUsage(
	snap geminiInteractionUsageSnapshot,
	result *service.ForwardResult,
	batchKey string,
	component string,
) error {
	doneCh := make(chan error, 1)
	h.submitMandatoryUsageRecordTask(snap.parentCtx, func(ctx context.Context) {
		completed := false
		defer func() {
			if !completed {
				doneCh <- errors.New("gemini batch usage task aborted before completion")
			}
		}()
		recCtx, cancel := context.WithTimeout(ctx, geminiExtOpTimeout)
		defer cancel()
		err := h.gatewayService.RecordUsage(recCtx, &service.RecordUsageInput{
			Result:             result,
			APIKey:             snap.apiKey,
			User:               snap.apiKey.User,
			Account:            snap.account,
			Subscription:       snap.subscription,
			InboundEndpoint:    snap.inboundEndpoint,
			UpstreamEndpoint:   snap.upstreamEndpoint,
			ChannelUsageFields: snap.channelUsageFields,
			RequestPayloadHash: snap.requestPayloadHash,
			PricingAt:          snap.pricingAt,
			ForceCacheBilling:  snap.forceCacheBilling,
			APIKeyService:      h.apiKeyService,
			UserAgent:          snap.userAgent,
		})
		if err != nil {
			logger.LegacyPrintf(component, "record gemini batch usage failed (batch=%s): %v", batchKey, err)
			relCtx, relCancel := context.WithTimeout(context.Background(), geminiExtOpTimeout)
			h.geminiCompatService.ReleaseGeminiBatchBilled(relCtx, batchKey)
			relCancel()
		}
		completed = true
		doneCh <- err
	})
	return <-doneCh
}

func (h *GatewayHandler) settleGeminiBatchFromJSON(
	ctx context.Context,
	account *service.Account,
	meta *service.GeminiBatchMeta,
	batchJSON []byte,
	currentAPIKey *service.APIKey,
	currentSub *service.UserSubscription,
	component string,
) (terminal bool) {
	if h == nil || h.geminiCompatService == nil || account == nil || meta == nil || meta.BatchKey == "" {
		return false
	}
	settled, err := h.geminiCompatService.IsGeminiBatchSettled(ctx, meta.BatchKey)
	if err == nil && settled {
		h.geminiCompatService.RemoveGeminiBatchPending(ctx, meta.BatchKey)
		return true
	}

	usage, state, isTerminal, extractErr := h.geminiCompatService.ExtractGeminiBatchUsage(ctx, account, meta.UserID, batchJSON)
	if extractErr != nil {
		logger.LegacyPrintf(component, "extract gemini batch usage failed (batch=%s): %v", meta.BatchKey, extractErr)
		return false
	}
	if !isTerminal {
		return false
	}
	if usage == nil && state == "BATCH_STATE_SUCCEEDED" {
		// A successful batch may expose its usage asynchronously. Marking it as
		// settled now would permanently lose billing for its completed requests.
		return false
	}
	if usage == nil || !hasBillableGeminiUsage(*usage) {
		_ = h.geminiCompatService.MarkGeminiBatchSettled(ctx, meta.BatchKey)
		h.geminiCompatService.RemoveGeminiBatchPending(ctx, meta.BatchKey)
		return true
	}

	claimed, claimErr := h.geminiCompatService.ClaimGeminiBatchBilled(ctx, meta.BatchKey)
	if claimErr != nil || !claimed {
		return false
	}

	snap := buildGeminiBatchUsageSnapshotFromMeta(meta)
	snap.account = account

	// 水合完整 APIKey（含 User 与 Group 倍率），避免 RecordUsage 因 User 为空跳过记账或无法扣费。
	var billingAPIKey *service.APIKey
	if currentAPIKey != nil && (meta.APIKeyID == 0 || currentAPIKey.ID == meta.APIKeyID) {
		billingAPIKey = currentAPIKey
	} else if h.apiKeyService != nil && meta.APIKeyID > 0 {
		loadedKey, keyErr := h.apiKeyService.GetByID(ctx, meta.APIKeyID)
		if keyErr == nil && loadedKey != nil {
			billingAPIKey = loadedKey
		} else if errors.Is(keyErr, service.ErrAPIKeyNotFound) {
			h.geminiCompatService.ReleaseGeminiBatchBilled(ctx, meta.BatchKey)
			_ = h.geminiCompatService.MarkGeminiBatchSettled(ctx, meta.BatchKey)
			h.geminiCompatService.RemoveGeminiBatchPending(ctx, meta.BatchKey)
			return true
		}
	}
	if billingAPIKey != nil {
		if billingAPIKey.User == nil && currentAPIKey != nil && currentAPIKey.User != nil && billingAPIKey.UserID == currentAPIKey.User.ID {
			billingAPIKey.User = currentAPIKey.User
		}
		if billingAPIKey.Group == nil && currentAPIKey != nil && currentAPIKey.Group != nil &&
			billingAPIKey.GroupID != nil && currentAPIKey.GroupID != nil && *billingAPIKey.GroupID == *currentAPIKey.GroupID {
			billingAPIKey.Group = currentAPIKey.Group
		}
		if billingAPIKey.User == nil && meta.UserID > 0 {
			billingAPIKey.User = &service.User{ID: meta.UserID}
		}
		snap.apiKey = billingAPIKey
	} else if snap.apiKey != nil && snap.apiKey.User == nil && meta.UserID > 0 {
		snap.apiKey.User = &service.User{ID: meta.UserID}
	}

	if meta.SubscriptionID != nil {
		if currentSub != nil && currentSub.ID == *meta.SubscriptionID {
			snap.subscription = currentSub
		} else {
			sub, subErr := h.resolveGeminiInteractionSubscription(ctx, &service.GeminiInteractionMeta{
				APIKeyID:       meta.APIKeyID,
				UserID:         meta.UserID,
				SubscriptionID: meta.SubscriptionID,
			}, nil)
			if subErr != nil {
				h.geminiCompatService.ReleaseGeminiBatchBilled(ctx, meta.BatchKey)
				return false
			}
			snap.subscription = sub
		}
	}

	res := &service.ForwardResult{
		RequestID:     "batch:" + meta.BatchKey,
		Usage:         *usage,
		Model:         meta.OriginalModel,
		UpstreamModel: meta.MappedModel,
		CostFactor:    0.5, // Gemini Batch API 官方 5 折定价
	}
	if recErr := h.runMandatoryGeminiBatchRecordUsage(snap, res, meta.BatchKey, component); recErr != nil {
		return false
	}
	_ = h.geminiCompatService.MarkGeminiBatchSettled(ctx, meta.BatchKey)
	h.geminiCompatService.RemoveGeminiBatchPending(ctx, meta.BatchKey)
	return true
}

func (h *GatewayHandler) handleGeminiBatchCreate(c *gin.Context, modelName, action string) {
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
		"handler.gemini_v1beta.batch_create",
		zap.Int64("user_id", subject.UserID),
		zap.Int64("api_key_id", apiKey.ID),
		zap.Any("group_id", apiKey.GroupID),
		zap.String("model", modelName),
		zap.String("action", action),
	)
	if !h.checkGeminiGroupModelAllowlist(c, apiKey, modelName) {
		return
	}

	body, err := pkghttputil.ReadRequestBodyWithPrealloc(c.Request)
	if err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			googleError(c, http.StatusRequestEntityTooLarge, buildBodyTooLargeMessage(maxErr.Limit))
			return
		}
		googleError(c, http.StatusBadRequest, "Failed to read request body")
		return
	}

	if decision := h.checkSecurityAudit(c, reqLog, apiKey, subject, "gemini", modelName, body); decision != nil && !decision.AllowNextStage {
		googleSecurityAuditError(c, decision)
		return
	}

	subscription := h.resolveGeminiExtSubscription(c, apiKey)
	if h.billingCacheService != nil && apiKey.User != nil {
		if err := h.billingCacheService.CheckBillingEligibility(c.Request.Context(), apiKey.User, apiKey, apiKey.Group, subscription, service.PlatformGemini); err != nil {
			status, _, message, _ := billingErrorDetails(err)
			googleError(c, status, message)
			return
		}
	}

	inflightRelease, inflightErr := reserveInflightBalance(c, h.billingCacheService, h.gatewayService, apiKey, subscription, tokenInflightEstimate(modelName, body))
	if inflightErr != nil {
		status, _, message, _ := billingErrorDetails(inflightErr)
		googleError(c, status, message)
		return
	}
	defer inflightRelease()

	var boundAccountID int64
	if refKind, refID := service.ExtractGeminiExtReferencedResource(body); refKind != "" && refID != "" {
		accID, found, blocked := h.resolvePinnedGeminiExtAccount(c, reqLog, refKind, refID, subject.UserID, true)
		if blocked {
			return
		}
		if found {
			boundAccountID = accID
		}
	}

	ensureCompositeTargetPlatform(c, apiKey, modelName)
	if !compositeTargetPlatformAllowed(c, apiKey, modelName, service.PlatformGemini) {
		googleError(c, http.StatusNotFound, "Model is not routed to Gemini platform in this composite group")
		return
	}
	origModel := modelName
	if upstreamModel, ok := service.ResolvedUpstreamModelFromContext(c.Request.Context()); ok && upstreamModel != "" {
		modelName = upstreamModel
	}
	channelMapping, _ := h.gatewayService.ResolveChannelMappingAndRestrict(c.Request.Context(), apiKey.GroupID, modelName)
	if channelMapping.Mapped && channelMapping.MappedModel != "" {
		modelName = channelMapping.MappedModel
	}
	if geminiImageGenerationRejected(c, apiKey, body, origModel, modelName) {
		return
	}
	channelFields := channelMapping.ToUsageFields(origModel, modelName)

	userRelease, err := h.concurrencyHelper.AcquireUserSlotWithWait(c, subject.UserID, subject.Concurrency, false, nil)
	if err != nil {
		h.handleConcurrencyError(c, err, "user", false)
		return
	}
	defer userRelease()

	sessionHash := ""
	if boundAccountID == 0 {
		sessionHash = "gemini:ext:user:" + strconv.FormatInt(subject.UserID, 10)
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
	streamStarted := false

	for {
		var selection *service.AccountSelectionResult
		var selErr error
		var account *service.Account
		if boundAccountID > 0 {
			pinnedAcc, ok := h.loadValidatedPinnedGeminiAccount(c.Request.Context(), apiKey, boundAccountID)
			if !ok {
				googleError(c, http.StatusServiceUnavailable, "Pinned Gemini AI Studio account for batch input file is unavailable")
				return
			}
			account = pinnedAcc
		} else {
			selection, selErr = h.gatewayService.SelectAccountWithLoadAwareness(
				c.Request.Context(), apiKey.GroupID, sessionHash, modelName, excludedIDs, "", subject.UserID,
			)
			if selErr != nil || selection == nil || selection.Account == nil {
				switch fs.HandleSelectionExhausted(c.Request.Context()) {
				case FailoverContinue:
					continue
				case FailoverCanceled:
					return
				default:
					if fs.LastFailoverErr != nil {
						h.handleGeminiFailoverExhausted(c, fs.LastFailoverErr)
					} else {
						markOpsRoutingCapacityLimited(c)
						googleError(c, http.StatusServiceUnavailable, "No available Gemini AI Studio accounts")
					}
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
				googleError(c, http.StatusServiceUnavailable, "No available Gemini AI Studio accounts")
				return
			}
			continue
		}
		account = latest

		var accountRelease func()
		if selection != nil {
			rel, acquired := h.acquireGeminiEndpointAccountSlot(c, reqLog, geminiConcurrency, selection, false, &streamStarted)
			if !acquired {
				return
			}
			accountRelease = rel
		} else {
			rel, acquired, slotErr := geminiConcurrency.TryAcquireAccountSlot(c.Request.Context(), account.ID, account.Concurrency)
			if slotErr != nil {
				h.handleConcurrencyError(c, slotErr, "account", false)
				return
			}
			if !acquired {
				googleError(c, http.StatusTooManyRequests, "Pinned upstream account concurrency limit reached")
				return
			}
			accountRelease = rel
		}

		mappedModel := account.GetMappedModel(modelName)
		_, respBody, forwardErr := h.geminiCompatService.ForwardGeminiBatchEnqueue(
			c.Request.Context(), c, account, modelName, mappedModel, action, c.Request.URL.Query(), body, subject.UserID,
		)
		accountRelease()

		if forwardErr == nil {
			batchName := strings.TrimSpace(gjson.GetBytes(respBody, "name").String())
			batchKey := service.NormalizeGeminiBatchKey(batchName)
			if batchKey != "" {
				var subID *int64
				if subscription != nil && subscription.ID > 0 {
					id := subscription.ID
					subID = &id
				}
				meta := service.GeminiBatchMeta{
					BatchKey:           batchKey,
					BatchName:          batchName,
					AccountID:          account.ID,
					UserID:             subject.UserID,
					APIKeyID:           apiKey.ID,
					GroupID:            derefGroupID(apiKey.GroupID),
					SubscriptionID:     subID,
					OriginalModel:      modelName,
					MappedModel:        mappedModel,
					ChannelUsageFields: channelFields,
					InboundEndpoint:    "/v1beta/models/" + modelName + ":" + action,
					UpstreamEndpoint:   "/v1beta/models/" + mappedModel + ":" + action,
					UserAgent:          c.GetHeader("User-Agent"),
					CreatedAtUnix:      time.Now().Unix(),
				}
				metaCtx, cancelMeta := context.WithTimeout(context.Background(), geminiExtOpTimeout)
				_ = h.geminiCompatService.SaveGeminiBatchMeta(metaCtx, batchKey, meta)
				h.geminiCompatService.BindGeminiExtResourceAccount(metaCtx, "batch", batchKey, account.ID, subject.UserID, 7*24*time.Hour)
				if mappedModel != modelName && strings.HasPrefix(batchName, "models/") {
					if _, sub, ok := strings.Cut(strings.TrimPrefix(batchName, "models/"), "/"); ok {
						if opID, isOp := strings.CutPrefix(sub, "operations/"); isOp && service.IsSafeGeminiExtID(opID) {
							origBatchKey := service.NormalizeGeminiBatchKey("models/" + modelName + "/operations/" + opID)
							if origBatchKey != "" && origBatchKey != batchKey {
								_ = h.geminiCompatService.SaveGeminiBatchMeta(metaCtx, origBatchKey, meta)
								h.geminiCompatService.BindGeminiExtResourceAccount(metaCtx, "batch", origBatchKey, account.ID, subject.UserID, 7*24*time.Hour)
							}
						}
					}
				}
				if !h.settleGeminiBatchFromJSON(metaCtx, account, &meta, respBody, apiKey, subscription, "handler.gemini_v1beta.batch_create") {
					h.geminiCompatService.ScheduleGeminiBatchPending(metaCtx, batchKey, time.Now().Add(30*time.Second).UnixMilli())
					h.startGeminiInteractionPendingSweeper()
				}
				cancelMeta()
			}
			return
		}

		var failoverErr *service.UpstreamFailoverError
		if errors.As(forwardErr, &failoverErr) && boundAccountID == 0 {
			switch fs.HandleFailoverError(c.Request.Context(), h.gatewayService, account.ID, account.Platform, account.GetPoolModeRetryCount(), failoverErr) {
			case FailoverContinue:
				excludedIDs[account.ID] = struct{}{}
				continue
			case FailoverExhausted:
				h.handleGeminiFailoverExhausted(c, fs.LastFailoverErr)
				return
			case FailoverCanceled:
				return
			}
		}
		if failoverErr != nil {
			h.handleGeminiFailoverExhausted(c, failoverErr)
			return
		}
		return
	}
}

func (h *GatewayHandler) GeminiV1BetaBatchesList(c *gin.Context) {
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:     "handler.gemini_v1beta.batches.list",
		Method:        http.MethodGet,
		UpstreamPath:  "/v1beta/batches",
		BindKind:      "batch",
		ListFields:    []string{"operations", "batches"},
		UseUserSticky: true,
	})
}

func (h *GatewayHandler) settleGeminiBatchBeforeDelete(
	c *gin.Context,
	batchID string,
	apiKey *service.APIKey,
	subscription *service.UserSubscription,
) (blocked bool) {
	if h == nil || h.geminiCompatService == nil || batchID == "" {
		return false
	}
	subject, _ := middleware2.GetAuthSubjectFromContext(c)
	reqLog := requestLogger(c, "handler.gemini_v1beta.batch.pre_delete", zap.String("batch_id", batchID))
	accID, _, blocked := h.resolvePinnedGeminiExtAccount(c, reqLog, "batch", batchID, subject.UserID, true)
	if blocked {
		return true
	}
	settleCtx, cancel := context.WithTimeout(c.Request.Context(), 20*time.Second)
	defer cancel()
	settled, err := h.geminiCompatService.IsGeminiBatchSettled(settleCtx, batchID)
	if err == nil && settled {
		return false
	}
	meta, hasMeta := h.geminiCompatService.GetGeminiBatchMeta(settleCtx, batchID)
	if !hasMeta || meta == nil {
		return false
	}
	targetAccID := meta.AccountID
	if targetAccID <= 0 {
		targetAccID = accID
	}
	account, ok := h.loadValidatedPinnedGeminiAccount(settleCtx, apiKey, targetAccID)
	if !ok {
		return false
	}
	batchJSON, statusCode, fetchErr := h.geminiCompatService.FetchGeminiBatchSnapshot(settleCtx, account, meta.BatchName)
	if fetchErr == nil && statusCode >= 200 && statusCode < 400 {
		h.settleGeminiBatchFromJSON(settleCtx, account, meta, batchJSON, apiKey, subscription, "handler.gemini_v1beta.batch.delete")
	}
	return false
}

func (h *GatewayHandler) GeminiV1BetaBatchAction(c *gin.Context) {
	rawAction := strings.TrimPrefix(strings.TrimSpace(c.Param("batchAction")), "/")
	batchID, action, ok := service.ParseGeminiExtIDAction(rawAction, "cancel", "updateGenerateContentBatch", "updateEmbedContentBatch")
	if !ok {
		googleError(c, http.StatusBadRequest, "Invalid batch action path")
		return
	}
	if c.Request.Method == http.MethodDelete && action == "" {
		apiKey, _ := middleware2.GetAPIKeyFromContext(c)
		sub := h.resolveGeminiExtSubscription(c, apiKey)
		if h.settleGeminiBatchBeforeDelete(c, batchID, apiKey, sub) {
			return
		}
	}
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:    "handler.gemini_v1beta.batch",
		Method:       c.Request.Method,
		UpstreamPath: "/v1beta/batches/" + rawAction,
		ReadBody:     c.Request.Method == http.MethodPost || c.Request.Method == http.MethodPatch,
		BindKind:     "batch",
		PinKind:      "batch",
		PinID:        batchID,
		OnJSONSuccess: func(_ context.Context, apiKey *service.APIKey, _ middleware2.AuthSubject, subscription *service.UserSubscription, account *service.Account, _ string, _ []byte, respBody []byte, _ *service.ForwardResult) {
			if c.Request.Method != http.MethodGet || action != "" || h.geminiCompatService == nil {
				return
			}
			settleCtx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			if meta, hasMeta := h.geminiCompatService.GetGeminiBatchMeta(settleCtx, batchID); hasMeta && meta != nil {
				h.settleGeminiBatchFromJSON(settleCtx, account, meta, respBody, apiKey, subscription, "handler.gemini_v1beta.batch.get")
			}
		},
	})
}

func (h *GatewayHandler) GeminiV1BetaModelOperationGet(c *gin.Context) {
	modelName := strings.TrimSpace(c.Param("model"))
	opID := strings.TrimPrefix(strings.TrimSpace(c.Param("operationName")), "/")
	if !service.IsSafeGeminiModelPathSegment(modelName) || !service.IsSafeGeminiExtID(opID) {
		googleError(c, http.StatusBadRequest, "Invalid model operation path")
		return
	}
	batchKey := service.NormalizeGeminiBatchKey("models/" + modelName + "/operations/" + opID)
	pinKind, pinID := "model_op", modelName+"/"+opID
	if h.geminiCompatService != nil {
		if _, _, found := h.geminiCompatService.ResolveGeminiExtResource(c.Request.Context(), "batch", batchKey); found {
			pinKind, pinID = "batch", batchKey
		}
	}
	h.executeGeminiExtRequest(c, geminiExtRequestOpts{
		Component:    "handler.gemini_v1beta.model_operation",
		Method:       http.MethodGet,
		UpstreamPath: "/v1beta/models/" + modelName + "/operations/" + opID,
		UpstreamPathFn: func(mapped string) string {
			if mapped == "" {
				mapped = modelName
			}
			return "/v1beta/models/" + mapped + "/operations/" + opID
		},
		ExplicitModel: modelName,
		BindKind:      "model_op",
		PinKind:       pinKind,
		PinID:         pinID,
		OnJSONSuccess: func(_ context.Context, apiKey *service.APIKey, _ middleware2.AuthSubject, subscription *service.UserSubscription, account *service.Account, _ string, _ []byte, respBody []byte, _ *service.ForwardResult) {
			if h.geminiCompatService == nil {
				return
			}
			settleCtx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			if meta, hasMeta := h.geminiCompatService.GetGeminiBatchMeta(settleCtx, batchKey); hasMeta && meta != nil {
				h.settleGeminiBatchFromJSON(settleCtx, account, meta, respBody, apiKey, subscription, "handler.gemini_v1beta.model_operation")
			}
		},
	})
}

func (h *GatewayHandler) reconcilePendingGeminiBatch(ctx context.Context, batchKey string) {
	if h == nil || h.geminiCompatService == nil || h.gatewayService == nil || batchKey == "" {
		return
	}
	checkCtx, cancelCheck := context.WithTimeout(ctx, geminiExtOpTimeout)
	settled, _ := h.geminiCompatService.IsGeminiBatchSettled(checkCtx, batchKey)
	meta, hasMeta := h.geminiCompatService.GetGeminiBatchMeta(checkCtx, batchKey)
	cancelCheck()

	if settled || !hasMeta || meta == nil || meta.AccountID <= 0 {
		remCtx, cancelRem := context.WithTimeout(ctx, geminiExtOpTimeout)
		h.geminiCompatService.RemoveGeminiBatchPending(remCtx, batchKey)
		cancelRem()
		return
	}
	// 批处理最长超过 7 天仍未终态则退出轮询避免永久积压。
	if meta.CreatedAtUnix > 0 && time.Since(time.Unix(meta.CreatedAtUnix, 0)) > 7*24*time.Hour {
		remCtx, cancelRem := context.WithTimeout(ctx, geminiExtOpTimeout)
		h.geminiCompatService.RemoveGeminiBatchPending(remCtx, batchKey)
		cancelRem()
		return
	}

	acctCtx, cancelAcct := context.WithTimeout(ctx, geminiExtOpTimeout)
	account, acctErr := h.geminiCompatService.GetAccountByID(acctCtx, meta.AccountID)
	cancelAcct()
	if acctErr != nil || account == nil {
		return
	}

	fetchCtx, cancelFetch := context.WithTimeout(ctx, 20*time.Second)
	batchJSON, statusCode, fetchErr := h.geminiCompatService.FetchGeminiBatchSnapshot(fetchCtx, account, meta.BatchName)
	cancelFetch()
	if fetchErr != nil || statusCode >= 500 {
		schedCtx, cancelSched := context.WithTimeout(ctx, geminiExtOpTimeout)
		h.geminiCompatService.ScheduleGeminiBatchPending(schedCtx, batchKey, service.BuildGeminiBatchNextCheckAt(time.Now(), meta.CreatedAtUnix))
		cancelSched()
		return
	}
	if statusCode == http.StatusNotFound || statusCode == http.StatusGone {
		remCtx, cancelRem := context.WithTimeout(ctx, geminiExtOpTimeout)
		_ = h.geminiCompatService.MarkGeminiBatchSettled(remCtx, batchKey)
		h.geminiCompatService.RemoveGeminiBatchPending(remCtx, batchKey)
		cancelRem()
		return
	}
	if statusCode >= 400 {
		return
	}

	settleCtx, cancelSettle := context.WithTimeout(ctx, 45*time.Second)
	done := h.settleGeminiBatchFromJSON(settleCtx, account, meta, batchJSON, nil, nil, "handler.gemini_v1beta.batch.sweep")
	cancelSettle()
	if !done {
		schedCtx, cancelSched := context.WithTimeout(ctx, geminiExtOpTimeout)
		h.geminiCompatService.ScheduleGeminiBatchPending(schedCtx, batchKey, service.BuildGeminiBatchNextCheckAt(time.Now(), meta.CreatedAtUnix))
		cancelSched()
	}
}
