package service

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/geminicli"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/util/responseheaders"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// GeminiInteractionBillingRequestIDPrefix identifies interaction usage_logs rows
// so resolveUsageBillingRequestID preserves them instead of overwriting with client/local IDs.
const GeminiInteractionBillingRequestIDPrefix = "gemini_interaction:"

const (
	maxGeminiInteractionIDLen           = 256
	maxGeminiInteractionRequestIDLen    = 64
	geminiInteractionBindingTTL         = 7 * 24 * time.Hour
	geminiInteractionClaimLeaseTTL      = 2 * time.Minute
	geminiInteractionOwnerCachePrefix   = "gemini:interaction:owner:"
	geminiInteractionAccountCachePrefix = "gemini:interaction:acct:"
	geminiInteractionBilledClaimPrefix  = "gemini:interaction:billed:"
	geminiInteractionMetaCachePrefix    = "gemini:interaction:meta:"
	geminiInteractionSettledPrefix      = "gemini:interaction:settled:"
)

// GeminiInteractionBillingRequestID builds a deterministic <=64-char request_id
// for usage_logs (VARCHAR(64)), preserving the literal suffix for short IDs and
// hashing longer Google interaction IDs with a 176-bit SHA-256 hex digest.
func GeminiInteractionBillingRequestID(interactionID string) string {
	id := strings.TrimSpace(interactionID)
	if id == "" {
		return ""
	}
	candidate := GeminiInteractionBillingRequestIDPrefix + id
	if len(candidate) <= maxGeminiInteractionRequestIDLen {
		return candidate
	}
	sum := sha256.Sum256([]byte(id))
	return GeminiInteractionBillingRequestIDPrefix + hex.EncodeToString(sum[:22])
}

// GeminiInteractionMeta preserves creation-time billing and channel mapping metadata
// so deferred background GET polling and delayed reconciliation bill identically to synchronous responses.
type GeminiInteractionMeta struct {
	RequestModel       string             `json:"request_model,omitempty"`
	UpstreamModel      string             `json:"upstream_model,omitempty"`
	ChannelUsageFields ChannelUsageFields `json:"channel_usage_fields,omitempty"`
	PricingAtUnixNano  int64              `json:"pricing_at_unix_nano,omitempty"`
	ForceCacheBilling  bool               `json:"force_cache_billing,omitempty"`
	APIKeyID           int64              `json:"api_key_id,omitempty"`
	UserID             int64              `json:"user_id,omitempty"`
	AccountID          int64              `json:"account_id,omitempty"`
	SubscriptionID     *int64             `json:"subscription_id,omitempty"`
	RequestPayloadHash string             `json:"request_payload_hash,omitempty"`
	InboundEndpoint    string             `json:"inbound_endpoint,omitempty"`
	UpstreamEndpoint   string             `json:"upstream_endpoint,omitempty"`
	QuotaPlatform      string             `json:"quota_platform,omitempty"`
	UserAgent          string             `json:"user_agent,omitempty"`
	ClientIP           string             `json:"client_ip,omitempty"`
	SessionID          string             `json:"session_id,omitempty"`
}

// GeminiInteractionPendingIndex persists unresolved interaction IDs across
// reconcile deadlines and server restarts.
type GeminiInteractionPendingIndex interface {
	UpsertGeminiInteractionPending(ctx context.Context, interactionID string, nextCheckAtUnixMilli int64, ttl time.Duration) error
	ListDueGeminiInteractionPending(ctx context.Context, nowUnixMilli int64, limit int64) ([]string, error)
	RemoveGeminiInteractionPending(ctx context.Context, interactionID string) error
}

// IsSafeGeminiInteractionID reports whether id can be safely interpolated into an
// upstream /v1beta/interactions/{id} path segment without changing URL structure.
func IsSafeGeminiInteractionID(id string) bool {
	if id == "" || len(id) > maxGeminiInteractionIDLen {
		return false
	}
	dotsOnly := true
	for i := 0; i < len(id); i++ {
		if !isSafeUpstreamPathSegmentByte(id[i]) {
			return false
		}
		if id[i] != '.' {
			dotsOnly = false
		}
	}
	return !dotsOnly
}

// IsGeminiAIStudioEndpointAccountType checks platform/type eligibility for AI Studio-only
// endpoints (/v1beta/interactions, /v1beta/decisions) on scheduling snapshots.
func IsGeminiAIStudioEndpointAccountType(account *Account) bool {
	if account == nil || account.Platform != PlatformGemini {
		return false
	}
	if account.Type == AccountTypeAPIKey {
		return true
	}
	if account.Type == AccountTypeOAuth {
		// If credentials are present on the snapshot, exclude CLI/Code Assist/Google One OAuth early.
		if len(account.Credentials) > 0 {
			return isGeminiAIStudioOAuthAccount(account)
		}
		return true
	}
	return false
}

// IsGeminiAIStudioEndpointAccount verifies full hydrated account eligibility for
// AI Studio-only endpoints (/v1beta/interactions, /v1beta/decisions).
func IsGeminiAIStudioEndpointAccount(account *Account) bool {
	if !IsGeminiAIStudioEndpointAccountType(account) {
		return false
	}
	if account.Type == AccountTypeAPIKey {
		return strings.TrimSpace(account.GetCredential("api_key")) != ""
	}
	return isGeminiAIStudioOAuthAccount(account)
}

func isGeminiAIStudioOAuthAccount(account *Account) bool {
	if account == nil || account.Platform != PlatformGemini || account.Type != AccountTypeOAuth {
		return false
	}
	if account.IsGeminiCodeAssist() || account.IsGeminiGoogleOne() {
		return false
	}
	oauthType := strings.ToLower(strings.TrimSpace(account.GeminiOAuthType()))
	return oauthType == "ai_studio" || oauthType == ""
}

// GeminiAIStudioEndpointExclusions excludes accounts that cannot serve AI Studio-only
// endpoints, or excludes every account other than boundAccountID when boundAccountID > 0.
func (s *GatewayService) GeminiAIStudioEndpointExclusions(ctx context.Context, groupID *int64, boundAccountID int64) (map[int64]struct{}, error) {
	accounts, _, err := s.listSchedulableAccounts(ctx, groupID, PlatformGemini, false)
	if err != nil {
		return nil, err
	}
	excluded := make(map[int64]struct{}, len(accounts))
	for i := range accounts {
		account := &accounts[i]
		keep := IsGeminiAIStudioEndpointAccountType(account)
		if boundAccountID > 0 {
			keep = keep && account.ID == boundAccountID
		}
		if !keep {
			excluded[account.ID] = struct{}{}
		}
	}
	return excluded, nil
}

// ExtractGeminiAIStudioBodyModel extracts the requested model identifier from an AI Studio
// endpoint JSON payload, falling back to `agent` on /v1beta/interactions requests.
func ExtractGeminiAIStudioBodyModel(routePath string, body []byte) string {
	if !gjson.ValidBytes(body) {
		return ""
	}
	if model := strings.TrimSpace(gjson.GetBytes(body, "model").String()); model != "" {
		return model
	}
	if strings.Contains(routePath, "/interactions") {
		return strings.TrimSpace(gjson.GetBytes(body, "agent").String())
	}
	return ""
}

// BindGeminiInteractionAccount records the owning user and upstream account for an
// interaction ID so follow-up GET / cancel / DELETE / previous_interaction_id calls
// stay pinned to the originating account and isolated per user.
func (s *GeminiMessagesCompatService) BindGeminiInteractionAccount(
	ctx context.Context,
	groupID *int64,
	userID int64,
	interactionID string,
	accountID int64,
) error {
	if s == nil || s.cache == nil || !IsSafeGeminiInteractionID(interactionID) || accountID <= 0 {
		return nil
	}
	gid := derefGroupID(groupID)
	if userID > 0 {
		if err := s.cache.SetSessionAccountID(ctx, 0, geminiInteractionOwnerCachePrefix+interactionID, userID, geminiInteractionBindingTTL); err != nil {
			return err
		}
	}
	return s.cache.SetSessionAccountID(ctx, gid, geminiInteractionAccountCachePrefix+interactionID, accountID, geminiInteractionBindingTTL)
}

// ResolveGeminiInteractionAccount verifies interaction ownership and resolves the
// bound upstream account ID for follow-up GET / cancel / DELETE / previous_interaction_id calls.
// Returns verified=true only when both owner (if userID > 0) and account bindings exist and match.
func (s *GeminiMessagesCompatService) ResolveGeminiInteractionAccount(
	ctx context.Context,
	groupID *int64,
	userID int64,
	interactionID string,
) (accountID int64, verified bool, err error) {
	if !IsSafeGeminiInteractionID(interactionID) {
		return 0, false, nil
	}
	if s == nil || s.cache == nil {
		return 0, false, errors.New("interaction binding cache unavailable")
	}
	if userID > 0 {
		ownerID, ownerErr := s.cache.GetSessionAccountID(ctx, 0, geminiInteractionOwnerCachePrefix+interactionID)
		if ownerErr != nil {
			if errors.Is(ownerErr, ErrStickySessionNotFound) {
				return 0, false, nil
			}
			return 0, false, ownerErr
		}
		if ownerID <= 0 || ownerID != userID {
			return 0, false, nil
		}
	}
	acctID, acctErr := s.cache.GetSessionAccountID(ctx, derefGroupID(groupID), geminiInteractionAccountCachePrefix+interactionID)
	if acctErr != nil {
		if errors.Is(acctErr, ErrStickySessionNotFound) {
			return 0, false, nil
		}
		return 0, false, acctErr
	}
	if acctID <= 0 {
		return 0, false, nil
	}
	return acctID, true, nil
}

// ClaimGeminiInteractionBilled acquires a short recoverable processing lease for interactionID
// after verifying that settlement has not already completed.
func (s *GeminiMessagesCompatService) ClaimGeminiInteractionBilled(ctx context.Context, interactionID string) (bool, error) {
	if s == nil || s.cache == nil || !IsSafeGeminiInteractionID(interactionID) {
		return true, nil
	}
	settled, err := s.IsGeminiInteractionSettled(ctx, interactionID)
	if err != nil {
		return false, err
	}
	if settled {
		return false, nil
	}
	return s.cache.ClaimGrokVideoBilled(ctx, geminiInteractionBilledClaimPrefix+interactionID, geminiInteractionClaimLeaseTTL)
}

// MarkGeminiInteractionSettled records that RecordUsage completed for interactionID so
// background reconcilers can stop polling immediately.
func (s *GeminiMessagesCompatService) MarkGeminiInteractionSettled(ctx context.Context, interactionID string) error {
	if s == nil || !IsSafeGeminiInteractionID(interactionID) {
		return nil
	}
	_ = s.RemoveGeminiInteractionPending(ctx, interactionID)
	if s.cache == nil {
		return nil
	}
	return s.cache.SetGrokVideoPendingBilling(ctx, geminiInteractionSettledPrefix+interactionID, []byte("1"), geminiInteractionBindingTTL)
}

// IsGeminiInteractionSettled reports whether RecordUsage has completed for interactionID.
func (s *GeminiMessagesCompatService) IsGeminiInteractionSettled(ctx context.Context, interactionID string) (bool, error) {
	if s == nil || s.cache == nil || !IsSafeGeminiInteractionID(interactionID) {
		return false, nil
	}
	raw, err := s.cache.GetGrokVideoPendingBilling(ctx, geminiInteractionSettledPrefix+interactionID)
	if err != nil {
		return false, err
	}
	return len(raw) > 0, nil
}

// ReleaseGeminiInteractionBilled clears a billing claim so a failed RecordUsage can retry.
func (s *GeminiMessagesCompatService) ReleaseGeminiInteractionBilled(ctx context.Context, interactionID string) error {
	if s == nil || s.cache == nil || !IsSafeGeminiInteractionID(interactionID) {
		return nil
	}
	return s.cache.ReleaseGrokVideoBilled(ctx, geminiInteractionBilledClaimPrefix+interactionID)
}

// SaveGeminiInteractionMeta stores creation-time billing metadata for background polling.
func (s *GeminiMessagesCompatService) SaveGeminiInteractionMeta(ctx context.Context, interactionID string, meta GeminiInteractionMeta) error {
	if s == nil || s.cache == nil || !IsSafeGeminiInteractionID(interactionID) {
		return nil
	}
	payload, err := json.Marshal(meta)
	if err != nil || len(payload) == 0 {
		return err
	}
	return s.cache.SetGrokVideoPendingBilling(ctx, geminiInteractionMetaCachePrefix+interactionID, payload, geminiInteractionBindingTTL)
}

// GetGeminiInteractionMeta retrieves creation-time billing metadata if present.
func (s *GeminiMessagesCompatService) GetGeminiInteractionMeta(ctx context.Context, interactionID string) (*GeminiInteractionMeta, bool, error) {
	if s == nil || s.cache == nil || !IsSafeGeminiInteractionID(interactionID) {
		return nil, false, nil
	}
	raw, err := s.cache.GetGrokVideoPendingBilling(ctx, geminiInteractionMetaCachePrefix+interactionID)
	if err != nil {
		return nil, false, err
	}
	if len(raw) == 0 {
		return nil, false, nil
	}
	var meta GeminiInteractionMeta
	if err := json.Unmarshal(raw, &meta); err != nil {
		return nil, false, err
	}
	return &meta, true, nil
}

// ScheduleGeminiInteractionPending adds or updates interactionID in the pending billing queue.
func (s *GeminiMessagesCompatService) ScheduleGeminiInteractionPending(ctx context.Context, interactionID string, nextCheckAt time.Time) error {
	if s == nil || !IsSafeGeminiInteractionID(interactionID) {
		return nil
	}
	if idx, ok := s.cache.(GeminiInteractionPendingIndex); ok {
		return idx.UpsertGeminiInteractionPending(ctx, interactionID, nextCheckAt.UnixMilli(), geminiInteractionBindingTTL)
	}
	s.geminiInteractionPendingMu.Lock()
	if s.geminiInteractionPendingDue == nil {
		s.geminiInteractionPendingDue = make(map[string]time.Time)
	}
	s.geminiInteractionPendingDue[interactionID] = nextCheckAt
	s.geminiInteractionPendingMu.Unlock()
	return nil
}

// ListDueGeminiInteractionPending returns up to limit interaction IDs due for billing check at or before now.
func (s *GeminiMessagesCompatService) ListDueGeminiInteractionPending(ctx context.Context, now time.Time, limit int64) ([]string, error) {
	if s == nil {
		return nil, nil
	}
	if limit <= 0 {
		limit = 50
	}
	if idx, ok := s.cache.(GeminiInteractionPendingIndex); ok {
		return idx.ListDueGeminiInteractionPending(ctx, now.UnixMilli(), limit)
	}
	type pendingEntry struct {
		id  string
		due time.Time
	}
	s.geminiInteractionPendingMu.Lock()
	dueList := make([]pendingEntry, 0, len(s.geminiInteractionPendingDue))
	for id, due := range s.geminiInteractionPendingDue {
		if !due.After(now) {
			dueList = append(dueList, pendingEntry{id: id, due: due})
		}
	}
	s.geminiInteractionPendingMu.Unlock()
	sort.Slice(dueList, func(i, j int) bool {
		if !dueList[i].due.Equal(dueList[j].due) {
			return dueList[i].due.Before(dueList[j].due)
		}
		return dueList[i].id < dueList[j].id
	})
	if int64(len(dueList)) > limit {
		dueList = dueList[:limit]
	}
	out := make([]string, len(dueList))
	for i := range dueList {
		out[i] = dueList[i].id
	}
	return out, nil
}

// RemoveGeminiInteractionPending deletes interactionID from the pending billing queue.
func (s *GeminiMessagesCompatService) RemoveGeminiInteractionPending(ctx context.Context, interactionID string) error {
	if s == nil || !IsSafeGeminiInteractionID(interactionID) {
		return nil
	}
	if idx, ok := s.cache.(GeminiInteractionPendingIndex); ok {
		return idx.RemoveGeminiInteractionPending(ctx, interactionID)
	}
	s.geminiInteractionPendingMu.Lock()
	delete(s.geminiInteractionPendingDue, interactionID)
	s.geminiInteractionPendingMu.Unlock()
	return nil
}

// GetAccountByID returns an upstream account by ID for durable pending interaction reconciliation.
func (s *GeminiMessagesCompatService) GetAccountByID(ctx context.Context, accountID int64) (*Account, error) {
	if s == nil || s.accountRepo == nil || accountID <= 0 {
		return nil, errors.New("account repository unavailable")
	}
	return s.accountRepo.GetByID(ctx, accountID)
}

// ApplyGeminiInteractionMetaModels restores creation-time request and upstream mapped models
// onto a polling ForwardResult while preserving upstream's reported model in UpstreamResponseModel.
func ApplyGeminiInteractionMetaModels(res *ForwardResult, observedModel string, meta *GeminiInteractionMeta) {
	if res == nil {
		return
	}
	observedModel = strings.TrimSpace(observedModel)
	if res.UpstreamResponseModel == "" && observedModel != "" {
		res.UpstreamResponseModel = observedModel
	}
	if meta != nil {
		if reqModel := strings.TrimSpace(meta.RequestModel); reqModel != "" {
			res.Model = reqModel
		}
		if upstreamModel := strings.TrimSpace(meta.UpstreamModel); upstreamModel != "" {
			res.UpstreamModel = upstreamModel
		}
	}
	if res.Model == "" && observedModel != "" {
		res.Model = observedModel
	}
	if res.UpstreamModel == "" && observedModel != "" {
		res.UpstreamModel = observedModel
	}
	if res.ImageCount == 0 && (isImageGenerationModel(res.Model) || isImageGenerationModel(res.UpstreamModel)) &&
		(res.Usage.InputTokens > 0 || res.Usage.OutputTokens > 0 || res.Usage.CacheReadInputTokens > 0 || res.Usage.ImageOutputTokens > 0) {
		res.ImageCount = 1
		res.ImageEndpoint = true
	}
}

// ForwardDecisions proxies POST /v1beta/decisions to Google AI Studio compatible upstream.
func (s *GeminiMessagesCompatService) ForwardDecisions(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	originalModel, mappedModel string,
	body []byte,
) (*ForwardResult, error) {
	upstreamBody := rewriteGeminiAIStudioBodyModel(body, mappedModel)
	res, _, _, err := s.forwardGeminiAIStudioEndpoint(ctx, c, account, geminiAIStudioForwardSpec{
		Method:        http.MethodPost,
		Path:          "/v1beta/decisions",
		Body:          upstreamBody,
		OriginalModel: originalModel,
		MappedModel:   mappedModel,
		ExtractUsage:  extractGeminiInteractionUsage,
	})
	return res, err
}

// ForwardInteractionCreate proxies POST /v1beta/interactions (streaming or non-streaming).
// If onInteractionID is non-nil, it is invoked synchronously as soon as an interaction ID
// is observed in the upstream payload, before delivering that payload to the client.
func (s *GeminiMessagesCompatService) ForwardInteractionCreate(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	originalModel, mappedModel string,
	stream bool,
	body []byte,
	onInteractionID func(string),
) (*ForwardResult, string, error) {
	bodyModel := ExtractGeminiAIStudioBodyModel("/v1beta/interactions", body)
	if geminiImageGenerationDenied(c, body, bodyModel, originalModel, mappedModel) {
		return nil, "", s.writeGoogleError(c, http.StatusBadRequest, ImageGenerationPermissionMessage())
	}
	upstreamBody := rewriteGeminiAIStudioBodyModel(body, mappedModel)
	res, interactionID, _, err := s.forwardGeminiAIStudioEndpoint(ctx, c, account, geminiAIStudioForwardSpec{
		Method:          http.MethodPost,
		Path:            "/v1beta/interactions",
		Body:            upstreamBody,
		Stream:          stream,
		OriginalModel:   originalModel,
		MappedModel:     mappedModel,
		ExtractUsage:    extractGeminiInteractionUsage,
		OnInteractionID: onInteractionID,
	})
	return res, interactionID, err
}

// ForwardInteractionByID proxies GET/DELETE /v1beta/interactions/{id} and
// POST /v1beta/interactions/{id}/cancel.
func (s *GeminiMessagesCompatService) ForwardInteractionByID(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	method, interactionID string,
	isCancel bool,
	rawQuery url.Values,
	stream bool,
) (*ForwardResult, string, error) {
	if !IsSafeGeminiInteractionID(interactionID) {
		return nil, "", s.writeGoogleError(c, http.StatusBadRequest, "Invalid interaction ID")
	}
	path := "/v1beta/interactions/" + interactionID
	if isCancel {
		path += "/cancel"
	}
	res, _, observedModel, err := s.forwardGeminiAIStudioEndpoint(ctx, c, account, geminiAIStudioForwardSpec{
		Method:       method,
		Path:         path,
		Query:        sanitizedGeminiInteractionQuery(rawQuery),
		Stream:       stream,
		ExtractUsage: extractGeminiInteractionUsage,
	})
	return res, observedModel, err
}

type geminiAIStudioForwardSpec struct {
	Method          string
	Path            string
	Query           url.Values
	Body            []byte
	Stream          bool
	OriginalModel   string
	MappedModel     string
	ExtractUsage    func([]byte) *ClaudeUsage
	OnInteractionID func(string)
}

func (s *GeminiMessagesCompatService) forwardGeminiAIStudioEndpoint(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	spec geminiAIStudioForwardSpec,
) (*ForwardResult, string, string, error) {
	startTime := time.Now()
	if account == nil {
		return nil, "", "", s.writeGoogleError(c, http.StatusBadGateway, "No account selected")
	}

	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}

	buildReq := func(reqCtx context.Context) (*http.Request, error) {
		return s.buildGeminiAIStudioEndpointRequest(reqCtx, account, spec)
	}

	var resp *http.Response
	var checkedPolicy *ErrorPolicyResult
	maxAttempts := s.geminiMaxAttemptsPerAccount()
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		checkedPolicy = nil
		upstreamReq, err := buildReq(ctx)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return nil, "", "", err
			}
			return nil, "", "", s.writeGoogleError(c, http.StatusBadGateway, err.Error())
		}

		resp, err = s.httpUpstream.Do(upstreamReq, proxyURL, account.ID, account.Concurrency)
		if err != nil {
			return nil, "", "", s.handleUpstreamTransportError(ctx, c, account, err)
		}

		var checked *ErrorPolicyResult
		checked, resp = s.checkErrorPolicyInLoop(ctx, account, resp, spec.MappedModel)
		if checked != nil && *checked != ErrorPolicyNone {
			checkedPolicy = checked
			break
		}

		if resp.StatusCode >= 400 && s.shouldRetryGeminiUpstreamError(account, resp.StatusCode) {
			respBody := s.readUpstreamErrorBody(resp)
			_ = resp.Body.Close()

			if attempt < maxAttempts {
				upstreamMsg := sanitizeUpstreamErrorMessage(strings.TrimSpace(extractUpstreamErrorMessage(respBody)))
				appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
					ProxyID:            opsUpstreamProxyID(account),
					ProxyName:          opsUpstreamProxyName(account),
					Platform:           account.Platform,
					AccountID:          account.ID,
					AccountName:        account.Name,
					UpstreamStatusCode: resp.StatusCode,
					UpstreamRequestID:  resp.Header.Get("x-request-id"),
					Kind:               "retry",
					Message:            upstreamMsg,
					Detail:             s.upstreamErrorDetail(respBody),
				})
				logger.LegacyPrintf("service.gemini_messages_compat", "Gemini AI Studio endpoint account %d: upstream status %d, retry %d/%d", account.ID, resp.StatusCode, attempt, maxAttempts)
				sleepGeminiBackoff(attempt)
				continue
			}

			resp = &http.Response{
				StatusCode: resp.StatusCode,
				Header:     resp.Header.Clone(),
				Body:       io.NopCloser(bytes.NewReader(respBody)),
			}
			break
		}
		break
	}
	defer func() { _ = resp.Body.Close() }()

	requestID := resp.Header.Get("x-request-id")
	if requestID == "" {
		requestID = resp.Header.Get("x-goog-request-id")
	}
	if requestID != "" {
		c.Header("x-request-id", requestID)
	}

	if resp.StatusCode >= 400 {
		respBody := s.readUpstreamErrorBody(resp)
		if s.rateLimitService != nil {
			policy := s.finalErrorPolicy(ctx, account, resp.StatusCode, respBody, spec.MappedModel, checkedPolicy)
			switch policy {
			case ErrorPolicySkipped:
				if failoverErr := s.skippedErrorPolicyFailoverError(c, account, resp.StatusCode, respBody, requestID); failoverErr != nil {
					return nil, "", "", failoverErr
				}
				if account.IsCustomErrorCodesEnabled() {
					return nil, "", "", s.writeGeminiCustomCodeSkippedError(c, account, resp.StatusCode, requestID, respBody, func() {
						_ = s.writeGoogleError(c, http.StatusInternalServerError, geminiCustomCodeSkippedClientMessage)
					})
				}
				return nil, "", "", s.writeGeminiNativeUpstreamError(c, account, resp, respBody, requestID, false)
			case ErrorPolicyMatched, ErrorPolicyTempUnscheduled:
				if policy == ErrorPolicyMatched {
					s.handleGeminiUpstreamError(ctx, account, resp.StatusCode, resp.Header, respBody)
				}
				upstreamMsg := sanitizeUpstreamErrorMessage(strings.TrimSpace(extractUpstreamErrorMessage(respBody)))
				appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
					ProxyID:            opsUpstreamProxyID(account),
					ProxyName:          opsUpstreamProxyName(account),
					Platform:           account.Platform,
					AccountID:          account.ID,
					AccountName:        account.Name,
					UpstreamStatusCode: resp.StatusCode,
					UpstreamRequestID:  requestID,
					Kind:               "failover",
					Message:            upstreamMsg,
					Detail:             s.upstreamErrorDetail(respBody),
				})
				return nil, "", "", &UpstreamFailoverError{StatusCode: resp.StatusCode, ResponseBody: respBody}
			}
		}

		s.handleGeminiUpstreamError(ctx, account, resp.StatusCode, resp.Header, respBody)
		if s.shouldFailoverGeminiUpstreamError(resp.StatusCode) {
			upstreamMsg := sanitizeUpstreamErrorMessage(strings.TrimSpace(extractUpstreamErrorMessage(respBody)))
			appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
				ProxyID:            opsUpstreamProxyID(account),
				ProxyName:          opsUpstreamProxyName(account),
				Platform:           account.Platform,
				AccountID:          account.ID,
				AccountName:        account.Name,
				UpstreamStatusCode: resp.StatusCode,
				UpstreamRequestID:  requestID,
				Kind:               "failover",
				Message:            upstreamMsg,
				Detail:             s.upstreamErrorDetail(respBody),
			})
			return nil, "", "", &UpstreamFailoverError{StatusCode: resp.StatusCode, ResponseBody: respBody}
		}
		return nil, "", "", s.writeGeminiNativeUpstreamError(c, account, resp, respBody, requestID, false)
	}

	isSSE, bodyReader := classifyGeminiAIStudioResponseSSE(resp.Body, spec.Stream, resp.Header.Get("Content-Type"))
	var (
		usage               *ClaudeUsage
		firstTokenMs        *int
		clientDisconnect    bool
		interactionID       string
		observedModel       string
		imageCount          int
		interactionTerminal bool
	)
	if isSSE {
		streamRes, streamID, streamModel, streamImages, streamTerminal, streamErr := s.handleGeminiAIStudioStreamingResponse(c, resp, bodyReader, startTime, account, spec.ExtractUsage, spec.OnInteractionID)
		if streamErr != nil {
			return nil, streamID, streamModel, streamErr
		}
		usage = streamRes.usage
		firstTokenMs = streamRes.firstTokenMs
		clientDisconnect = streamRes.clientDisconnect
		interactionID = streamID
		observedModel = streamModel
		imageCount = streamImages
		interactionTerminal = streamTerminal
	} else {
		respBody, readErr := ReadUpstreamResponseBody(bodyReader, s.cfg, c, openAITooLargeError)
		if readErr != nil {
			return nil, "", "", readErr
		}
		interactionID = extractGeminiInteractionID(respBody)
		if interactionID != "" && spec.OnInteractionID != nil {
			spec.OnInteractionID(interactionID)
		}
		if s.responseHeaderFilter != nil {
			responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
		}
		contentType := resp.Header.Get("Content-Type")
		if contentType == "" {
			contentType = "application/json"
		}
		MarkResponseCommitted(c)
		c.Data(resp.StatusCode, contentType, respBody)

		observedModel = extractGeminiInteractionModel(respBody)
		observer := upstreamResponseModelObserverFromContext(c)
		if observer == nil {
			observer = beginUpstreamResponseModelObservation(c)
		}
		observeGeminiInteractionModel(observer, respBody, true)
		imageCount = countGeminiInteractionImages(respBody)
		_, isTerminalStatus, hasStatus := extractGeminiInteractionTerminalStatus(respBody)
		if hasStatus {
			interactionTerminal = isTerminalStatus
		} else if spec.Method == http.MethodPost {
			// Synchronous creates/decisions without an explicit status field complete inline.
			interactionTerminal = true
		}
		if spec.ExtractUsage != nil {
			usage = spec.ExtractUsage(respBody)
		}
	}

	if usage == nil {
		usage = &ClaudeUsage{}
	}
	model := spec.OriginalModel
	if model == "" {
		model = observedModel
	}
	upstreamModel := spec.MappedModel
	if upstreamModel == "" {
		upstreamModel = observedModel
	}
	if imageCount == 0 && (isImageGenerationModel(model) || isImageGenerationModel(upstreamModel)) && (usage.InputTokens > 0 || usage.OutputTokens > 0 || usage.ImageOutputTokens > 0) {
		imageCount = 1
	}

	return &ForwardResult{
		RequestID:                     requestID,
		UpstreamHeaders:               resp.Header,
		Usage:                         *usage,
		Model:                         model,
		UpstreamModel:                 upstreamModel,
		UpstreamResponseModel:         observedUpstreamResponseModel(c),
		UpstreamResponseModelConflict: observedUpstreamResponseModelConflict(c),
		Stream:                        isSSE,
		Duration:                      time.Since(startTime),
		FirstTokenMs:                  firstTokenMs,
		ClientDisconnect:              clientDisconnect,
		ImageEndpoint:                 imageCount > 0,
		ImageCount:                    imageCount,
		InteractionTerminal:           interactionTerminal,
	}, interactionID, observedModel, nil
}

// FetchGeminiInteractionSnapshot queries GET /v1beta/interactions/{id} in the background
// (without a *gin.Context) so deferred Deep Research / background interactions can be
// billed accurately even if the client stops polling.
func (s *GeminiMessagesCompatService) FetchGeminiInteractionSnapshot(
	ctx context.Context,
	account *Account,
	interactionID string,
) (result *ForwardResult, status string, terminal bool, err error) {
	if s == nil || s.httpUpstream == nil || account == nil || !IsSafeGeminiInteractionID(interactionID) {
		return nil, "", false, errors.New("cannot fetch interaction snapshot")
	}
	req, err := s.buildGeminiAIStudioEndpointRequest(ctx, account, geminiAIStudioForwardSpec{
		Method: http.MethodGet,
		Path:   "/v1beta/interactions/" + interactionID,
	})
	if err != nil {
		return nil, "", false, err
	}
	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}
	resp, err := s.httpUpstream.Do(req, proxyURL, account.ID, account.Concurrency)
	if err != nil {
		return nil, "", false, err
	}
	defer func() { _ = resp.Body.Close() }()
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
	if err != nil {
		return nil, "", false, err
	}
	if resp.StatusCode >= 400 {
		return nil, "", resp.StatusCode == http.StatusNotFound, fmt.Errorf("upstream status %d", resp.StatusCode)
	}

	status, terminal, _ = extractGeminiInteractionTerminalStatus(respBody)

	usage := extractGeminiInteractionUsage(respBody)
	if usage == nil {
		usage = &ClaudeUsage{}
	}
	observedModel := extractGeminiInteractionModel(respBody)
	observer := &upstreamResponseModelObserver{}
	observeGeminiInteractionModel(observer, respBody, true)
	imageCount := countGeminiInteractionImages(respBody)
	if imageCount == 0 && isImageGenerationModel(observedModel) && (usage.InputTokens > 0 || usage.OutputTokens > 0 || usage.ImageOutputTokens > 0 || status == "completed") {
		imageCount = 1
	}

	return &ForwardResult{
		RequestID:                     resp.Header.Get("x-request-id"),
		UpstreamHeaders:               resp.Header,
		Usage:                         *usage,
		Model:                         observedModel,
		UpstreamModel:                 observedModel,
		UpstreamResponseModel:         observer.Model(),
		UpstreamResponseModelConflict: observer.Conflict(),
		ImageEndpoint:                 imageCount > 0,
		ImageCount:                    imageCount,
		InteractionTerminal:           terminal,
	}, status, terminal, nil
}

func (s *GeminiMessagesCompatService) buildGeminiAIStudioEndpointRequest(
	ctx context.Context,
	account *Account,
	spec geminiAIStudioForwardSpec,
) (*http.Request, error) {
	if !isAllowedGeminiAIStudioEndpointPath(spec.Path) {
		return nil, errors.New("invalid gemini ai studio endpoint path")
	}

	baseURL := account.GetGeminiBaseURL(geminicli.AIStudioBaseURL)
	normalizedBaseURL, err := s.validateUpstreamBaseURL(baseURL)
	if err != nil {
		return nil, err
	}
	fullURL := strings.TrimRight(normalizedBaseURL, "/") + spec.Path
	if encoded := spec.Query.Encode(); encoded != "" {
		fullURL += "?" + encoded
	}

	var bodyReader io.Reader
	if len(spec.Body) > 0 {
		bodyReader = bytes.NewReader(spec.Body)
	}
	req, err := http.NewRequestWithContext(ctx, spec.Method, fullURL, bodyReader)
	if err != nil {
		return nil, err
	}
	if len(spec.Body) > 0 {
		req.Header.Set("Content-Type", "application/json")
	}
	if spec.Stream {
		req.Header.Set("Accept", "text/event-stream")
	}

	switch account.Type {
	case AccountTypeAPIKey:
		apiKey := strings.TrimSpace(account.GetCredential("api_key"))
		if apiKey == "" {
			return nil, errors.New("gemini api_key not configured")
		}
		req.Header.Set("x-goog-api-key", apiKey)
	case AccountTypeOAuth:
		if s.tokenProvider == nil {
			return nil, errors.New("gemini token provider not configured")
		}
		accessToken, err := s.tokenProvider.GetAccessToken(ctx, account)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", "Bearer "+accessToken)
	default:
		return nil, fmt.Errorf("unsupported account type: %s", account.Type)
	}
	return req, nil
}

func isAllowedGeminiAIStudioEndpointPath(path string) bool {
	if path == "/v1beta/decisions" || path == "/v1beta/interactions" {
		return true
	}
	rest, ok := strings.CutPrefix(path, "/v1beta/interactions/")
	if !ok || rest == "" {
		return false
	}
	if id, isCancel := strings.CutSuffix(rest, "/cancel"); isCancel {
		return IsSafeGeminiInteractionID(id)
	}
	return IsSafeGeminiInteractionID(rest)
}

func sanitizedGeminiInteractionQuery(raw url.Values) url.Values {
	if len(raw) == 0 {
		return nil
	}
	out := url.Values{}
	if stream := strings.TrimSpace(raw.Get("stream")); stream != "" {
		if strings.EqualFold(stream, "true") || strings.EqualFold(stream, "false") {
			out.Set("stream", strings.ToLower(stream))
		}
	}
	if lastEventID := strings.TrimSpace(raw.Get("last_event_id")); lastEventID != "" {
		out.Set("last_event_id", lastEventID)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// classifyGeminiAIStudioResponseSSE decides whether to parse the upstream body as SSE.
// An explicit media type (such as application/json) overrides specStream; missing
// Content-Type with specStream peeks a bounded prefix to distinguish JSON from SSE.
func classifyGeminiAIStudioResponseSSE(body io.Reader, specStream bool, contentType string) (bool, io.Reader) {
	if mediaType, _, err := mime.ParseMediaType(contentType); err == nil && mediaType != "" {
		return strings.EqualFold(mediaType, "text/event-stream"), body
	}
	if !specStream {
		return false, body
	}
	br := bufio.NewReader(body)
	peek, _ := br.Peek(512)
	trimmed := bytes.TrimSpace(bytes.TrimPrefix(peek, []byte("\xef\xbb\xbf")))
	if len(trimmed) > 0 && (trimmed[0] == '{' || trimmed[0] == '[') {
		return false, br
	}
	return true, br
}

func (s *GeminiMessagesCompatService) handleGeminiAIStudioStreamingResponse(
	c *gin.Context,
	resp *http.Response,
	bodyReader io.Reader,
	startTime time.Time,
	account *Account,
	extractUsageFn func([]byte) *ClaudeUsage,
	onInteractionID func(string),
) (*geminiNativeStreamResult, string, string, int, bool, error) {
	if s.responseHeaderFilter != nil {
		responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	}
	c.Status(resp.StatusCode)
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "text/event-stream; charset=utf-8"
	}
	c.Header("Content-Type", contentType)

	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		return nil, "", "", 0, false, errors.New("streaming not supported")
	}

	if bodyReader == nil {
		bodyReader = resp.Body
	}
	reader := newBoundedSSELineReader(bodyReader, s.cfg)
	usage := &ClaudeUsage{}
	observer := upstreamResponseModelObserverFromContext(c)
	if observer == nil {
		observer = beginUpstreamResponseModelObservation(c)
	}
	var (
		firstTokenMs         *int
		clientDisconnect     bool
		interactionID        string
		observedModel        string
		snapshotImages       int
		unindexedDeltaImages int
		seenDeltaImageKeys   map[string]struct{}
		currentSSEEvent      string
		interactionTerminal  bool
	)

	for {
		line, err := reader.ReadLineString()
		if len(line) > 0 && (err == nil || errors.Is(err, io.EOF)) {
			trimmed := strings.TrimRight(line, "\r\n")
			if trimmed == "" {
				currentSSEEvent = ""
			} else if strings.HasPrefix(trimmed, "event:") {
				currentSSEEvent = strings.ToLower(strings.TrimSpace(strings.TrimPrefix(trimmed, "event:")))
			} else if strings.HasPrefix(trimmed, "data:") {
				payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
				if payload != "" && payload != "[DONE]" {
					rawBytes := []byte(payload)
					if id := extractGeminiInteractionID(rawBytes); id != "" {
						if id != interactionID && onInteractionID != nil {
							onInteractionID(id)
						}
						interactionID = id
					}
					if m := extractGeminiInteractionModel(rawBytes); m != "" {
						observedModel = m
					}
					if extractUsageFn != nil {
						if u := extractUsageFn(rawBytes); u != nil {
							usage = u
						}
					}
					if imgCount := countGeminiInteractionImages(rawBytes); imgCount > snapshotImages {
						snapshotImages = imgCount
					}
					if isGeminiInteractionImageDelta(rawBytes) {
						if key, hasKey := geminiInteractionDeltaImageKey(rawBytes); hasKey {
							if seenDeltaImageKeys == nil {
								seenDeltaImageKeys = make(map[string]struct{})
							}
							seenDeltaImageKeys[key] = struct{}{}
						} else {
							unindexedDeltaImages++
						}
					}
					eventType := strings.ToLower(strings.TrimSpace(gjson.GetBytes(rawBytes, "event_type").String()))
					if eventType == "" {
						eventType = currentSSEEvent
					}
					_, statusTerminal, hasStatus := extractGeminiInteractionTerminalStatus(rawBytes)
					switch {
					case hasStatus && statusTerminal:
						interactionTerminal = true
					case hasStatus && !statusTerminal:
						interactionTerminal = false
					case eventType == "interaction.completed" || eventType == "interaction.complete" || eventType == "interaction.failed" || eventType == "interaction.cancelled" || eventType == "interaction.canceled":
						interactionTerminal = true
					case eventType == "interaction.start" || eventType == "interaction.created":
						interactionTerminal = false
					}
					observeGeminiInteractionModel(observer, rawBytes, interactionTerminal)
					if firstTokenMs == nil {
						ms := int(time.Since(startTime).Milliseconds())
						firstTokenMs = &ms
					}
				}
			}
			_, _ = io.WriteString(c.Writer, line)
			flusher.Flush()
		}

		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			observedImages := max(snapshotImages, len(seenDeltaImageKeys)+unindexedDeltaImages)
			if !interactionTerminal || (!geminiStreamAbortedWithBillableOutput(c, err, usage) && observedImages == 0) {
				return nil, interactionID, observedModel, observedImages, interactionTerminal, err
			}
			clientDisconnect = true
			logger.LegacyPrintf("service.gemini_messages_compat", "[GeminiAPI] Client disconnected during completed interactions stream (account=%d), billing collected usage: %v", account.ID, err)
			break
		}
	}

	return &geminiNativeStreamResult{
		usage:            usage,
		firstTokenMs:     firstTokenMs,
		clientDisconnect: clientDisconnect,
	}, interactionID, observedModel, max(snapshotImages, len(seenDeltaImageKeys)+unindexedDeltaImages), interactionTerminal, nil
}

func rewriteGeminiAIStudioBodyModel(body []byte, mappedModel string) []byte {
	mappedModel = strings.TrimSpace(mappedModel)
	if mappedModel == "" || !gjson.ValidBytes(body) {
		return body
	}
	if model := gjson.GetBytes(body, "model"); model.Type == gjson.String {
		if current := strings.TrimSpace(model.String()); current != "" {
			if current == mappedModel {
				return body
			}
			if updated, err := sjson.SetBytes(body, "model", mappedModel); err == nil {
				return updated
			}
		}
	}
	if agent := gjson.GetBytes(body, "agent"); agent.Type == gjson.String {
		if current := strings.TrimSpace(agent.String()); current != "" {
			if current == mappedModel {
				return body
			}
			if updated, err := sjson.SetBytes(body, "agent", mappedModel); err == nil {
				return updated
			}
		}
	}
	return body
}

func extractGeminiInteractionID(data []byte) string {
	for _, path := range []string{"interaction.id", "interaction_id", "id"} {
		if id := strings.TrimSpace(gjson.GetBytes(data, path).String()); IsSafeGeminiInteractionID(id) {
			return id
		}
	}
	return ""
}

func extractGeminiInteractionModel(data []byte) string {
	for _, path := range []string{"interaction.model", "interaction.agent", "model", "agent", "modelVersion"} {
		if model := strings.TrimSpace(gjson.GetBytes(data, path).String()); model != "" {
			return model
		}
	}
	return ""
}

func observeGeminiInteractionModel(observer *upstreamResponseModelObserver, payload []byte, terminal bool) {
	if observer == nil || len(payload) == 0 {
		return
	}
	observer.ObserveGemini(payload)
	if model := extractGeminiInteractionModel(payload); model != "" {
		observer.Observe(model, terminal)
	}
}

func countGeminiInteractionImages(payload []byte) int {
	if len(payload) == 0 || !gjson.ValidBytes(payload) {
		return 0
	}
	count := countGeminiImageOutputs(payload)
	outputsCount := 0
	if outputs := gjson.GetBytes(payload, "interaction.outputs"); outputs.IsArray() {
		outputsCount = countInteractionOutputsArrayImages(outputs)
	} else if outputs := gjson.GetBytes(payload, "outputs"); outputs.IsArray() {
		outputsCount = countInteractionOutputsArrayImages(outputs)
	}
	stepsCount := 0
	if steps := gjson.GetBytes(payload, "interaction.steps"); steps.IsArray() {
		stepsCount = countInteractionStepsArrayImages(steps)
	} else if steps := gjson.GetBytes(payload, "steps"); steps.IsArray() {
		stepsCount = countInteractionStepsArrayImages(steps)
	}
	return count + max(outputsCount, stepsCount)
}

func countInteractionOutputsArrayImages(outputs gjson.Result) int {
	count := 0
	outputs.ForEach(func(_, item gjson.Result) bool {
		if isInteractionImageItem(item) {
			count++
		}
		return true
	})
	return count
}

func countInteractionStepsArrayImages(steps gjson.Result) int {
	count := 0
	steps.ForEach(func(_, step gjson.Result) bool {
		if !strings.EqualFold(strings.TrimSpace(step.Get("type").String()), "model_output") {
			return true
		}
		if content := step.Get("content"); content.IsArray() {
			count += countInteractionOutputsArrayImages(content)
		}
		return true
	})
	return count
}

func isGeminiInteractionImageDelta(payload []byte) bool {
	delta := gjson.GetBytes(payload, "delta")
	return delta.IsObject() && isInteractionImageItem(delta)
}

func geminiInteractionDeltaImageKey(payload []byte) (string, bool) {
	stepIdx := gjson.GetBytes(payload, "step_index")
	contentIdx := gjson.GetBytes(payload, "content_index")
	idx := gjson.GetBytes(payload, "index")
	if !stepIdx.Exists() && !contentIdx.Exists() && !idx.Exists() {
		return "", false
	}
	return stepIdx.Raw + ":" + contentIdx.Raw + ":" + idx.Raw, true
}

func extractGeminiInteractionTerminalStatus(payload []byte) (status string, isTerminal bool, hasStatus bool) {
	status = strings.ToLower(strings.TrimSpace(gjson.GetBytes(payload, "interaction.status").String()))
	if status == "" {
		status = strings.ToLower(strings.TrimSpace(gjson.GetBytes(payload, "status").String()))
	}
	if status == "" {
		return "", false, false
	}
	switch status {
	case "completed", "failed", "cancelled", "canceled", "incomplete", "requires_action", "requires_continuation", "budget_exceeded":
		return status, true, true
	default:
		return status, false, true
	}
}

func isInteractionImageItem(item gjson.Result) bool {
	if !strings.EqualFold(strings.TrimSpace(item.Get("type").String()), "image") {
		return false
	}
	return strings.TrimSpace(item.Get("data").String()) != "" ||
		strings.TrimSpace(item.Get("uri").String()) != "" ||
		strings.TrimSpace(item.Get("url").String()) != ""
}

func extractGeminiInteractionUsage(data []byte) *ClaudeUsage {
	for _, path := range []string{"interaction.usage", "metadata.total_usage", "total_usage", "usage"} {
		u := gjson.GetBytes(data, path)
		if !u.Exists() || !u.IsObject() {
			continue
		}
		totalInput := int(u.Get("total_input_tokens").Int())
		totalCached := int(u.Get("total_cached_tokens").Int())
		totalOutput := int(u.Get("total_output_tokens").Int())
		totalThoughts := int(u.Get("total_thought_tokens").Int())
		totalToolUse := int(u.Get("total_tool_use_tokens").Int())

		imageTokens := 0
		if modalities := u.Get("output_tokens_by_modality"); modalities.Exists() && modalities.IsArray() {
			modalities.ForEach(func(_, item gjson.Result) bool {
				if strings.EqualFold(item.Get("modality").String(), "IMAGE") {
					imageTokens = int(item.Get("tokens").Int())
					return false
				}
				return true
			})
		}

		// Interactions total_input_tokens reflects full prompt context; subtract cached tokens cleanly.
		inputTokens := max(totalInput-totalCached, 0) + totalToolUse
		outputTokens := totalOutput + totalThoughts
		if inputTokens > 0 || outputTokens > 0 || totalCached > 0 || imageTokens > 0 {
			return &ClaudeUsage{
				InputTokens:          inputTokens,
				OutputTokens:         outputTokens,
				CacheReadInputTokens: totalCached,
				ImageOutputTokens:    imageTokens,
			}
		}
	}
	return extractGeminiUsage(data)
}

// ParseGeminiInteractionBody validates JSON payload for POST /v1beta/interactions
// and returns (modelOrAgent, previousInteractionID, stream, ok).
func ParseGeminiInteractionBody(body []byte) (model string, previousInteractionID string, stream bool, ok bool) {
	model, previousInteractionID, stream, _, ok = ParseGeminiInteractionBodyWithStreamFlag(body)
	return model, previousInteractionID, stream, ok
}

// ParseGeminiInteractionBodyWithStreamFlag validates JSON payload for POST /v1beta/interactions
// and also reports whether the "stream" field was explicitly provided in JSON.
func ParseGeminiInteractionBodyWithStreamFlag(body []byte) (model string, previousInteractionID string, stream bool, hasStreamField bool, ok bool) {
	if !json.Valid(body) {
		return "", "", false, false, false
	}
	model = ExtractGeminiAIStudioBodyModel("/interactions", body)
	previousInteractionID = strings.TrimSpace(gjson.GetBytes(body, "previous_interaction_id").String())
	streamField := gjson.GetBytes(body, "stream")
	hasStreamField = streamField.Exists()
	stream = streamField.Bool()
	return model, previousInteractionID, stream, hasStreamField, true
}
