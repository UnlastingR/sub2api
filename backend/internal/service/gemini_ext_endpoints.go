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
	"net/http"
	"net/url"
	"path"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/util/responseheaders"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	geminiExtDefaultBaseURL = "https://generativelanguage.googleapis.com"

	geminiExtResourceTTL     = 90 * 24 * time.Hour
	geminiExtFileResourceTTL = 50 * time.Hour
	geminiUploadSessionTTL   = 50 * time.Hour
	geminiAuthTokenTTL       = 24 * time.Hour
	geminiBatchMetaTTL       = 7 * 24 * time.Hour
	geminiBatchBillClaimTTL  = 5 * time.Minute
	geminiBatchPendingTTL    = 7 * 24 * time.Hour

	geminiExtResKeyPrefix      = "gemini:ext:res:v1:"
	geminiUploadSessionPrefix  = "gemini:ext:upload:v1:"
	geminiAuthTokenPrefix      = "gemini:ext:authtoken:v1:"
	geminiBatchMetaPrefix      = "gemini:batch:meta:v1:"
	geminiBatchBillClaimPrefix = "gemini:batch:bill-claim:v1:"
	geminiBatchSettledPrefix   = "gemini:batch:settled:v1:"
	geminiBatchPendingPrefix   = "batch:"

	geminiStreamSmallRetryCap = 1 << 20  // 1 MiB: 可重放小请求/start 元数据上限
	geminiStreamJSONRespCap   = 4 << 20  // 4 MiB: 上传完成响应体 JSON 解析上限
	geminiBatchJSONLScanMax   = 32 << 20 // 32 MiB: 单行 JSONL 响应最大扫描长度
)

var geminiExtStrippedQueryKeys = map[string]struct{}{
	"key":            {},
	"api_key":        {},
	"x-goog-api-key": {},
	"access_token":   {},
	"auth_token":     {},
	"upload_id":      {},
}

var geminiExtPassThroughUploadHeaders = []string{
	"X-Goog-Upload-Status",
	"X-Goog-Upload-Size-Received",
	"X-Goog-Upload-Chunk-Granularity",
	"X-Goog-Upload-Command",
	"Content-Disposition",
	"Content-Range",
	"Accept-Ranges",
}

// GeminiExtResourceBinding records owning upstream account and gateway user for an external Gemini resource.
type GeminiExtResourceBinding struct {
	AccountID int64 `json:"account_id"`
	UserID    int64 `json:"user_id"`
}

// GeminiUploadSession stores resumable upload state keyed by gateway-issued upload_id.
type GeminiUploadSession struct {
	AccountID     int64  `json:"account_id"`
	UserID        int64  `json:"user_id"`
	UpstreamURL   string `json:"upstream_url"`
	TargetPath    string `json:"target_path,omitempty"`
	OriginalModel string `json:"original_model,omitempty"`
}

// GeminiAuthTokenMeta stores issuing gateway context for ephemeral Live API tokens.
type GeminiAuthTokenMeta struct {
	AccountID     int64  `json:"account_id"`
	UserID        int64  `json:"user_id"`
	APIKeyID      int64  `json:"api_key_id"`
	GroupID       int64  `json:"group_id"`
	Model         string `json:"model,omitempty"`
	UpstreamToken string `json:"upstream_token,omitempty"`
	ExpiresAt     int64  `json:"expires_at,omitempty"`
}

// GeminiBatchMeta captures batch enqueue attribution for asynchronous 50%-discounted settlement.
type GeminiBatchMeta struct {
	BatchKey           string             `json:"batch_key"`
	BatchName          string             `json:"batch_name"`
	AccountID          int64              `json:"account_id"`
	UserID             int64              `json:"user_id"`
	APIKeyID           int64              `json:"api_key_id"`
	GroupID            int64              `json:"group_id"`
	SubscriptionID     *int64             `json:"subscription_id,omitempty"`
	OriginalModel      string             `json:"original_model"`
	MappedModel        string             `json:"mapped_model"`
	ChannelUsageFields ChannelUsageFields `json:"channel_usage_fields,omitempty"`
	InboundEndpoint    string             `json:"inbound_endpoint,omitempty"`
	UpstreamEndpoint   string             `json:"upstream_endpoint,omitempty"`
	UserAgent          string             `json:"user_agent,omitempty"`
	CreatedAtUnix      int64              `json:"created_at_unix"`
}

// GeminiAIStudioExtJSONSpec describes a standard JSON proxy request to an AI Studio extension endpoint.
type GeminiAIStudioExtJSONSpec struct {
	Method                string
	Path                  string
	Query                 url.Values
	Body                  []byte
	OriginalModel         string
	MappedModel           string
	UserID                int64
	BindKind              string
	TransformResponseJSON func([]byte) ([]byte, error)
	OnJSON                func([]byte)
}

// GeminiAIStudioStreamSpec describes a streaming upload/download request to AI Studio.
type GeminiAIStudioStreamSpec struct {
	Method        string
	Path          string
	Query         url.Values
	UserID        int64
	BindKind      string
	IsDownload    bool
	UploadSession *GeminiUploadSession
	OriginalModel string
	MappedModel   string
}

// IsSafeGeminiExtID checks whether a resource ID is safe to interpolate into an upstream URL path segment.
func IsSafeGeminiExtID(id string) bool {
	return isSafeUpstreamPathSegment(strings.TrimSpace(id))
}

// IsSafeGeminiExtSubpath checks whether a multi-segment workspace file subpath is safe to forward upstream.
func IsSafeGeminiExtSubpath(subpath string) bool {
	trimmed := strings.Trim(strings.TrimSpace(subpath), "/")
	if trimmed == "" || len(trimmed) > 1024 {
		return false
	}
	if strings.ContainsAny(trimmed, "\\?#%\r\n") {
		return false
	}
	segments := strings.Split(trimmed, "/")
	if len(segments) > 32 {
		return false
	}
	for _, seg := range segments {
		if seg == "" || seg == "." || seg == ".." || len(seg) > 256 {
			return false
		}
		for i := 0; i < len(seg); i++ {
			b := seg[i]
			switch {
			case b >= 'a' && b <= 'z', b >= 'A' && b <= 'Z', b >= '0' && b <= '9':
			case b == '.', b == '_', b == '-', b == '+', b == '@':
			default:
				return false
			}
		}
	}
	return true
}

// ParseGeminiExtIDAction splits "{id}" or "{id}:{action}" and validates the ID and optional action allowlist.
func ParseGeminiExtIDAction(raw string, allowedActions ...string) (id string, action string, ok bool) {
	trimmed := strings.TrimSpace(strings.TrimPrefix(raw, "/"))
	if trimmed == "" || strings.Contains(trimmed, "/") {
		return "", "", false
	}
	if left, right, hasColon := strings.Cut(trimmed, ":"); hasColon {
		if !IsSafeGeminiExtID(left) || right == "" {
			return "", "", false
		}
		if len(allowedActions) > 0 && !slices.Contains(allowedActions, right) {
			return "", "", false
		}
		if len(allowedActions) == 0 && !IsSafeGeminiExtID(right) {
			return "", "", false
		}
		return left, right, true
	}
	if !IsSafeGeminiExtID(trimmed) {
		return "", "", false
	}
	return trimmed, "", true
}

// SanitizeGeminiExtQuery removes gateway credential/upload proxy parameters before forwarding upstream.
func SanitizeGeminiExtQuery(raw url.Values) url.Values {
	if len(raw) == 0 {
		return nil
	}
	out := make(url.Values, len(raw))
	for k, vals := range raw {
		if _, stripped := geminiExtStrippedQueryKeys[strings.ToLower(strings.TrimSpace(k))]; stripped {
			continue
		}
		for _, v := range vals {
			out.Add(k, v)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func isAllowedGeminiAIStudioExtensionPath(rawPath string) bool {
	p := strings.TrimSpace(rawPath)
	switch {
	case p == "/v1beta/agents",
		p == "/v1beta/environments",
		p == "/v1beta/credentials",
		p == "/v1beta/triggers",
		p == "/v1beta/webhooks",
		p == "/v1beta/auth_tokens",
		p == "/v1beta/files",
		p == "/v1beta/files:register",
		p == "/upload/v1beta/files",
		p == "/v1beta/fileSearchStores",
		p == "/v1beta/batches":
		return true
	case strings.HasPrefix(p, "/v1beta/agents/"):
		rest := strings.TrimPrefix(p, "/v1beta/agents/")
		return IsSafeGeminiExtID(rest)
	case strings.HasPrefix(p, "/v1beta/credentials/"):
		rest := strings.TrimPrefix(p, "/v1beta/credentials/")
		return IsSafeGeminiExtID(rest)
	case strings.HasPrefix(p, "/v1beta/webhooks/"):
		rest := strings.TrimPrefix(p, "/v1beta/webhooks/")
		_, _, ok := ParseGeminiExtIDAction(rest, "ping", "rotateSigningSecret")
		return ok
	case strings.HasPrefix(p, "/v1beta/environments/"):
		rest := strings.TrimPrefix(p, "/v1beta/environments/")
		if envID, _, ok := ParseGeminiExtIDAction(rest, "get", "delete"); ok {
			return envID != ""
		}
		envID, sub, hasSlash := strings.Cut(rest, "/")
		if !hasSlash || !IsSafeGeminiExtID(envID) {
			return false
		}
		if sub == "files" {
			return true
		}
		if filePath, ok := strings.CutPrefix(sub, "files/"); ok {
			return IsSafeGeminiExtSubpath(filePath)
		}
		return false
	case strings.HasPrefix(p, "/upload/v1beta/environments/"):
		rest := strings.TrimPrefix(p, "/upload/v1beta/environments/")
		envID, sub, hasSlash := strings.Cut(rest, "/")
		if !hasSlash || !IsSafeGeminiExtID(envID) {
			return false
		}
		if sub == "files" {
			return true
		}
		if filePath, ok := strings.CutPrefix(sub, "files/"); ok {
			return IsSafeGeminiExtSubpath(filePath)
		}
		return false
	case strings.HasPrefix(p, "/v1beta/triggers/"):
		rest := strings.TrimPrefix(p, "/v1beta/triggers/")
		if IsSafeGeminiExtID(rest) {
			return true
		}
		trigID, sub, hasSlash := strings.Cut(rest, "/")
		if !hasSlash || !IsSafeGeminiExtID(trigID) {
			return false
		}
		if sub == "executions" {
			return true
		}
		if execID, ok := strings.CutPrefix(sub, "executions/"); ok {
			return IsSafeGeminiExtID(execID)
		}
		return false
	case strings.HasPrefix(p, "/v1beta/files/"):
		rest := strings.TrimPrefix(p, "/v1beta/files/")
		_, _, ok := ParseGeminiExtIDAction(rest, "download")
		return ok
	case strings.HasPrefix(p, "/upload/v1beta/files/"):
		rest := strings.TrimPrefix(p, "/upload/v1beta/files/")
		return IsSafeGeminiExtID(rest)
	case strings.HasPrefix(p, "/download/v1beta/files/"):
		rest := strings.TrimPrefix(p, "/download/v1beta/files/")
		_, _, ok := ParseGeminiExtIDAction(rest, "download")
		return ok
	case strings.HasPrefix(p, "/v1beta/fileSearchStores/"):
		rest := strings.TrimPrefix(p, "/v1beta/fileSearchStores/")
		if _, _, ok := ParseGeminiExtIDAction(rest, "importFile", "uploadToFileSearchStore"); ok {
			return true
		}
		storeID, sub, hasSlash := strings.Cut(rest, "/")
		if !hasSlash || !IsSafeGeminiExtID(storeID) {
			return false
		}
		if sub == "documents" {
			return true
		}
		if docID, ok := strings.CutPrefix(sub, "documents/"); ok {
			return IsSafeGeminiExtID(docID)
		}
		if opID, ok := strings.CutPrefix(sub, "operations/"); ok {
			return IsSafeGeminiExtID(opID)
		}
		if opID, ok := strings.CutPrefix(sub, "upload/operations/"); ok {
			return IsSafeGeminiExtID(opID)
		}
		return false
	case strings.HasPrefix(p, "/upload/v1beta/fileSearchStores/"):
		rest := strings.TrimPrefix(p, "/upload/v1beta/fileSearchStores/")
		_, action, ok := ParseGeminiExtIDAction(rest, "uploadToFileSearchStore")
		return ok && action == "uploadToFileSearchStore"
	case strings.HasPrefix(p, "/v1beta/batches/"):
		rest := strings.TrimPrefix(p, "/v1beta/batches/")
		_, _, ok := ParseGeminiExtIDAction(rest, "cancel", "updateGenerateContentBatch", "updateEmbedContentBatch")
		return ok
	case strings.HasPrefix(p, "/v1beta/models/"):
		rest := strings.TrimPrefix(p, "/v1beta/models/")
		if modelPart, sub, hasSlash := strings.Cut(rest, "/"); hasSlash {
			if strings.Contains(modelPart, "..") || !IsSafeGeminiModelPathSegment(modelPart) {
				return false
			}
			if opID, ok := strings.CutPrefix(sub, "operations/"); ok {
				return IsSafeGeminiExtID(opID)
			}
			return false
		}
		modelPart, action, hasColon := strings.Cut(rest, ":")
		if !hasColon || strings.Contains(modelPart, "..") || !IsSafeGeminiModelPathSegment(modelPart) {
			return false
		}
		return action == "batchGenerateContent" || action == "asyncBatchEmbedContent" || action == "asyncBatchEmbedContents"
	default:
		return false
	}
}

func (s *GeminiMessagesCompatService) buildGeminiAIStudioExtURL(account *Account, endpointPath string, query url.Values) (string, error) {
	if !isAllowedGeminiAIStudioExtensionPath(endpointPath) {
		return "", errors.New("unsupported upstream endpoint path")
	}
	rawBase := geminiExtDefaultBaseURL
	if account != nil {
		rawBase = account.GetGeminiBaseURL(geminiExtDefaultBaseURL)
	}
	normalizedBase, err := s.validateUpstreamBaseURL(rawBase)
	if err != nil {
		return "", err
	}
	u, err := url.Parse(normalizedBase)
	if err != nil {
		return "", err
	}
	trimmedBasePath := strings.TrimRight(u.Path, "/")
	// 如果配置了带版本后缀的 base_url（如 .../v1beta），去掉版本尾巴以避免拼出 /v1beta/upload/v1beta/...
	for _, ver := range []string{"/v1beta", "/v1alpha", "/v1"} {
		if strings.HasSuffix(trimmedBasePath, ver) {
			trimmedBasePath = strings.TrimSuffix(trimmedBasePath, ver)
			break
		}
	}
	u.Path = trimmedBasePath + endpointPath
	sanitized := SanitizeGeminiExtQuery(query)
	if len(sanitized) > 0 {
		u.RawQuery = sanitized.Encode()
	} else {
		u.RawQuery = ""
	}
	return u.String(), nil
}

func geminiExtResourceTTLForKind(kind string) time.Duration {
	if kind == "file" {
		return geminiExtFileResourceTTL
	}
	return geminiExtResourceTTL
}

func geminiExtBindingCacheKey(kind, id string) string {
	return geminiExtResKeyPrefix + strings.TrimSpace(kind) + ":" + strings.TrimSpace(id)
}

// BindGeminiExtResourceAccount records account and owning user for a Gemini extension resource ID.
func (s *GeminiMessagesCompatService) BindGeminiExtResourceAccount(ctx context.Context, kind, id string, accountID, userID int64, ttl time.Duration) {
	if s == nil || s.cache == nil || accountID <= 0 {
		return
	}
	kind = strings.TrimSpace(kind)
	id = strings.TrimSpace(id)
	if kind == "" || id == "" {
		return
	}
	if ttl <= 0 {
		ttl = geminiExtResourceTTLForKind(kind)
	}
	raw, err := json.Marshal(GeminiExtResourceBinding{
		AccountID: accountID,
		UserID:    userID,
	})
	if err != nil {
		return
	}
	_ = s.cache.SetGrokVideoPendingBilling(ctx, geminiExtBindingCacheKey(kind, id), raw, ttl)
}

// HasGeminiExtCache reports whether Redis/cache backing is configured for extension resource isolation.
func (s *GeminiMessagesCompatService) HasGeminiExtCache() bool {
	return s != nil && s.cache != nil
}

// ResolveGeminiExtResource retrieves the pinned account and owning user for a resource ID.
func (s *GeminiMessagesCompatService) ResolveGeminiExtResource(ctx context.Context, kind, id string) (accountID int64, userID int64, ok bool) {
	if s == nil || s.cache == nil {
		return 0, 0, false
	}
	kind = strings.TrimSpace(kind)
	id = strings.TrimSpace(id)
	if kind == "" || id == "" {
		return 0, 0, false
	}
	raw, err := s.cache.GetGrokVideoPendingBilling(ctx, geminiExtBindingCacheKey(kind, id))
	if err != nil || len(raw) == 0 {
		return 0, 0, false
	}
	var binding GeminiExtResourceBinding
	if err := json.Unmarshal(raw, &binding); err != nil || binding.AccountID <= 0 {
		return 0, 0, false
	}
	return binding.AccountID, binding.UserID, true
}

// FilterGeminiExtListByUser strips items from an upstream collection response that are not
// explicitly owned by callerUserID when cache isolation is active.
func (s *GeminiMessagesCompatService) FilterGeminiExtListByUser(
	ctx context.Context,
	data []byte,
	listFields []string,
	defaultKind string,
	callerUserID int64,
) ([]byte, error) {
	if s == nil || s.cache == nil || callerUserID <= 0 || len(listFields) == 0 || len(data) == 0 || !gjson.ValidBytes(data) {
		return data, nil
	}
	updated := data
	for _, listField := range listFields {
		if listField == "" {
			continue
		}
		arr := gjson.GetBytes(updated, listField)
		if !arr.Exists() || !arr.IsArray() {
			continue
		}
		kept := make([]any, 0)
		arr.ForEach(func(_, item gjson.Result) bool {
			name := strings.TrimSpace(item.Get("name").String())
			kind, id := extractGeminiExtResourceKindID(name, defaultKind)
			if kind == "" || id == "" {
				return true
			}
			_, ownerUserID, ok := s.ResolveGeminiExtResource(ctx, kind, id)
			if ok && ownerUserID == callerUserID {
				var parsed any
				if json.Unmarshal([]byte(item.Raw), &parsed) == nil {
					kept = append(kept, parsed)
				}
			}
			return true
		})
		next, err := sjson.SetBytes(updated, listField, kept)
		if err != nil {
			return nil, err
		}
		updated = next
	}
	return updated, nil
}

func extractGeminiExtResourceKindID(name, defaultKind string) (string, string) {
	trimmed := strings.TrimSpace(strings.TrimPrefix(name, "/"))
	if trimmed == "" {
		return "", ""
	}
	if !strings.Contains(trimmed, "/") && defaultKind != "" && IsSafeGeminiExtID(trimmed) {
		return defaultKind, trimmed
	}
	for prefix, kind := range map[string]string{
		"agents/":           "agent",
		"environments/":     "env",
		"credentials/":      "cred",
		"triggers/":         "trigger",
		"webhooks/":         "webhook",
		"files/":            "file",
		"fileSearchStores/": "store",
		"batches/":          "batch",
	} {
		if id, ok := strings.CutPrefix(trimmed, prefix); ok {
			if left, _, hasSlash := strings.Cut(id, "/"); hasSlash {
				id = left
			}
			if IsSafeGeminiExtID(id) {
				return kind, id
			}
		}
	}
	return "", ""
}

// bindGeminiResourceByName maps canonical resource names (e.g. "files/abc", "batches/123") to cache bindings.
func (s *GeminiMessagesCompatService) bindGeminiResourceByName(ctx context.Context, resourceName string, accountID, userID int64) {
	name := strings.TrimSpace(strings.TrimPrefix(resourceName, "/"))
	if name == "" {
		return
	}
	switch {
	case strings.HasPrefix(name, "agents/"):
		if id := strings.TrimPrefix(name, "agents/"); IsSafeGeminiExtID(id) {
			s.BindGeminiExtResourceAccount(ctx, "agent", id, accountID, userID, geminiExtResourceTTL)
		}
	case strings.HasPrefix(name, "environments/"):
		if id := strings.TrimPrefix(name, "environments/"); IsSafeGeminiExtID(id) {
			s.BindGeminiExtResourceAccount(ctx, "env", id, accountID, userID, geminiExtResourceTTL)
		}
	case strings.HasPrefix(name, "credentials/"):
		if id := strings.TrimPrefix(name, "credentials/"); IsSafeGeminiExtID(id) {
			s.BindGeminiExtResourceAccount(ctx, "cred", id, accountID, userID, geminiExtResourceTTL)
		}
	case strings.HasPrefix(name, "triggers/"):
		if id := strings.TrimPrefix(name, "triggers/"); IsSafeGeminiExtID(id) {
			s.BindGeminiExtResourceAccount(ctx, "trigger", id, accountID, userID, geminiExtResourceTTL)
		}
	case strings.HasPrefix(name, "webhooks/"):
		if id := strings.TrimPrefix(name, "webhooks/"); IsSafeGeminiExtID(id) {
			s.BindGeminiExtResourceAccount(ctx, "webhook", id, accountID, userID, geminiExtResourceTTL)
		}
	case strings.HasPrefix(name, "files/"):
		if id := strings.TrimPrefix(name, "files/"); IsSafeGeminiExtID(id) {
			s.BindGeminiExtResourceAccount(ctx, "file", id, accountID, userID, geminiExtFileResourceTTL)
		}
	case strings.HasPrefix(name, "fileSearchStores/"):
		rest := strings.TrimPrefix(name, "fileSearchStores/")
		storeID, sub, hasSlash := strings.Cut(rest, "/")
		if !hasSlash {
			if IsSafeGeminiExtID(storeID) {
				s.BindGeminiExtResourceAccount(ctx, "store", storeID, accountID, userID, geminiExtResourceTTL)
			}
			return
		}
		if !IsSafeGeminiExtID(storeID) {
			return
		}
		s.BindGeminiExtResourceAccount(ctx, "store", storeID, accountID, userID, geminiExtResourceTTL)
		if opID, ok := strings.CutPrefix(sub, "operations/"); ok && IsSafeGeminiExtID(opID) {
			s.BindGeminiExtResourceAccount(ctx, "store_op", storeID+"/"+opID, accountID, userID, geminiExtResourceTTL)
		} else if opID, ok := strings.CutPrefix(sub, "upload/operations/"); ok && IsSafeGeminiExtID(opID) {
			s.BindGeminiExtResourceAccount(ctx, "store_upload_op", storeID+"/"+opID, accountID, userID, geminiExtResourceTTL)
		}
	case strings.HasPrefix(name, "batches/"):
		if id := strings.TrimPrefix(name, "batches/"); IsSafeGeminiExtID(id) {
			s.BindGeminiExtResourceAccount(ctx, "batch", id, accountID, userID, geminiBatchMetaTTL)
		}
	case strings.HasPrefix(name, "models/"):
		rest := strings.TrimPrefix(name, "models/")
		modelPart, sub, hasSlash := strings.Cut(rest, "/")
		if hasSlash && IsSafeGeminiModelPathSegment(modelPart) {
			if opID, ok := strings.CutPrefix(sub, "operations/"); ok && IsSafeGeminiExtID(opID) {
				s.BindGeminiExtResourceAccount(ctx, "model_op", modelPart+"/"+opID, accountID, userID, geminiBatchMetaTTL)
			}
		}
	}
}

// BindGeminiExtResourcesFromJSON extracts resource names from single-object JSON responses and pins ownership.
func (s *GeminiMessagesCompatService) BindGeminiExtResourcesFromJSON(ctx context.Context, data []byte, defaultKind string, accountID, userID int64) {
	if s == nil || s.cache == nil || accountID <= 0 || len(data) == 0 || !gjson.ValidBytes(data) {
		return
	}
	for _, field := range []string{"name", "file.name", "metadata.name", "response.name", "output.responsesFile", "response.responsesFile"} {
		if val := strings.TrimSpace(gjson.GetBytes(data, field).String()); val != "" {
			if strings.Contains(val, "/") {
				s.bindGeminiResourceByName(ctx, val, accountID, userID)
			} else if defaultKind != "" && IsSafeGeminiExtID(val) {
				s.BindGeminiExtResourceAccount(ctx, defaultKind, val, accountID, userID, geminiExtResourceTTLForKind(defaultKind))
			}
		}
	}
}

// ExtractGeminiExtReferencedResource resolves referenced existing resource IDs inside create/import request payloads.
func ExtractGeminiExtReferencedResource(body []byte) (kind, id string) {
	if len(body) == 0 || !gjson.ValidBytes(body) {
		return "", ""
	}
	for _, p := range []string{
		"batch.inputConfig.fileName",
		"batch.input_config.file_name",
		"inputConfig.fileName",
		"input_config.file_name",
		"fileName",
		"file_name",
		"file.name",
	} {
		val := strings.TrimSpace(gjson.GetBytes(body, p).String())
		if strings.HasPrefix(val, "files/") {
			if fileID := strings.TrimPrefix(val, "files/"); IsSafeGeminiExtID(fileID) {
				return "file", fileID
			}
		}
	}
	return "", ""
}

// SaveGeminiUploadSession stores a resumable upload session keyed by uploadID.
func (s *GeminiMessagesCompatService) SaveGeminiUploadSession(ctx context.Context, uploadID string, session GeminiUploadSession) error {
	if s == nil || s.cache == nil || !IsSafeGeminiExtID(uploadID) || session.AccountID <= 0 || session.UpstreamURL == "" {
		return errors.New("invalid upload session")
	}
	raw, err := json.Marshal(session)
	if err != nil {
		return err
	}
	return s.cache.SetGrokVideoPendingBilling(ctx, geminiUploadSessionPrefix+uploadID, raw, geminiUploadSessionTTL)
}

// GetGeminiUploadSession retrieves a saved resumable upload session by uploadID.
func (s *GeminiMessagesCompatService) GetGeminiUploadSession(ctx context.Context, uploadID string) (*GeminiUploadSession, bool) {
	if s == nil || s.cache == nil || !IsSafeGeminiExtID(uploadID) {
		return nil, false
	}
	raw, err := s.cache.GetGrokVideoPendingBilling(ctx, geminiUploadSessionPrefix+uploadID)
	if err != nil || len(raw) == 0 {
		return nil, false
	}
	var sess GeminiUploadSession
	if err := json.Unmarshal(raw, &sess); err != nil || sess.AccountID <= 0 || sess.UpstreamURL == "" {
		return nil, false
	}
	return &sess, true
}

func hashGeminiAuthTokenKey(rawToken string) string {
	normalized := strings.TrimSpace(rawToken)
	normalized = strings.TrimPrefix(normalized, "Token ")
	normalized = strings.TrimPrefix(normalized, "Bearer ")
	normalized = strings.TrimSpace(normalized)
	if normalized == "" {
		return ""
	}
	if !strings.HasPrefix(normalized, "auth_tokens/") {
		normalized = "auth_tokens/" + normalized
	}
	sum := sha256.Sum256([]byte(normalized))
	return hex.EncodeToString(sum[:])
}

// SaveGeminiAuthTokenMeta records gateway context for an upstream ephemeral Live API token.
func (s *GeminiMessagesCompatService) SaveGeminiAuthTokenMeta(ctx context.Context, rawToken string, meta GeminiAuthTokenMeta) error {
	hashed := hashGeminiAuthTokenKey(rawToken)
	if s == nil || s.cache == nil || hashed == "" || meta.AccountID <= 0 || meta.UserID <= 0 {
		return errors.New("invalid auth token metadata")
	}
	raw, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	ttl := geminiAuthTokenTTL
	if meta.ExpiresAt > 0 {
		until := time.Until(time.Unix(meta.ExpiresAt, 0)) + time.Minute
		if until > time.Minute && until < ttl {
			ttl = until
		}
	}
	return s.cache.SetGrokVideoPendingBilling(ctx, geminiAuthTokenPrefix+hashed, raw, ttl)
}

// GetGeminiAuthTokenMeta retrieves stored gateway attribution for a Live ephemeral token.
func (s *GeminiMessagesCompatService) GetGeminiAuthTokenMeta(ctx context.Context, rawToken string) (*GeminiAuthTokenMeta, bool) {
	hashed := hashGeminiAuthTokenKey(rawToken)
	if s == nil || s.cache == nil || hashed == "" {
		return nil, false
	}
	raw, err := s.cache.GetGrokVideoPendingBilling(ctx, geminiAuthTokenPrefix+hashed)
	if err != nil || len(raw) == 0 {
		return nil, false
	}
	var meta GeminiAuthTokenMeta
	if err := json.Unmarshal(raw, &meta); err != nil || meta.AccountID <= 0 || meta.UserID <= 0 {
		return nil, false
	}
	if meta.ExpiresAt > 0 && time.Now().Unix() > meta.ExpiresAt {
		return nil, false
	}
	return &meta, true
}

// NormalizeGeminiBatchKey builds a safe cache key suffix from a batch/operation resource name.
func NormalizeGeminiBatchKey(batchName string) string {
	trimmed := strings.TrimSpace(strings.TrimPrefix(batchName, "/"))
	if trimmed == "" {
		return ""
	}
	if id, ok := strings.CutPrefix(trimmed, "batches/"); ok && IsSafeGeminiExtID(id) {
		return id
	}
	if rest, ok := strings.CutPrefix(trimmed, "models/"); ok {
		modelPart, sub, hasSlash := strings.Cut(rest, "/")
		if hasSlash && IsSafeGeminiModelPathSegment(modelPart) {
			if opID, hasOp := strings.CutPrefix(sub, "operations/"); hasOp && IsSafeGeminiExtID(opID) {
				return "modelop_" + modelPart + "__" + opID
			}
		}
	}
	if IsSafeGeminiExtID(trimmed) {
		return trimmed
	}
	return ""
}

// SaveGeminiBatchMeta saves batch enqueue metadata for later 50%-discounted usage reconciliation.
func (s *GeminiMessagesCompatService) SaveGeminiBatchMeta(ctx context.Context, batchKey string, meta GeminiBatchMeta) error {
	if s == nil || s.cache == nil || batchKey == "" || meta.AccountID <= 0 {
		return errors.New("invalid gemini batch metadata")
	}
	raw, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	return s.cache.SetGrokVideoPendingBilling(ctx, geminiBatchMetaPrefix+batchKey, raw, geminiBatchMetaTTL)
}

// GetGeminiBatchMeta retrieves batch enqueue metadata by batchKey.
func (s *GeminiMessagesCompatService) GetGeminiBatchMeta(ctx context.Context, batchKey string) (*GeminiBatchMeta, bool) {
	if s == nil || s.cache == nil || batchKey == "" {
		return nil, false
	}
	raw, err := s.cache.GetGrokVideoPendingBilling(ctx, geminiBatchMetaPrefix+batchKey)
	if err != nil || len(raw) == 0 {
		return nil, false
	}
	var meta GeminiBatchMeta
	if err := json.Unmarshal(raw, &meta); err != nil || meta.AccountID <= 0 {
		return nil, false
	}
	return &meta, true
}

func (s *GeminiMessagesCompatService) ClaimGeminiBatchBilled(ctx context.Context, batchKey string) (bool, error) {
	if s == nil || s.cache == nil || batchKey == "" {
		return false, errors.New("gemini batch billing claim unavailable")
	}
	return s.cache.ClaimGrokVideoBilled(ctx, geminiBatchBillClaimPrefix+batchKey, geminiBatchBillClaimTTL)
}

func (s *GeminiMessagesCompatService) ReleaseGeminiBatchBilled(ctx context.Context, batchKey string) {
	if s == nil || s.cache == nil || batchKey == "" {
		return
	}
	_ = s.cache.ReleaseGrokVideoBilled(ctx, geminiBatchBillClaimPrefix+batchKey)
}

func (s *GeminiMessagesCompatService) MarkGeminiBatchSettled(ctx context.Context, batchKey string) error {
	if s == nil || s.cache == nil || batchKey == "" {
		return nil
	}
	return s.cache.SetGrokVideoPendingBilling(ctx, geminiBatchSettledPrefix+batchKey, []byte("1"), geminiBatchMetaTTL)
}

func (s *GeminiMessagesCompatService) IsGeminiBatchSettled(ctx context.Context, batchKey string) (bool, error) {
	if s == nil || s.cache == nil || batchKey == "" {
		return false, nil
	}
	raw, err := s.cache.GetGrokVideoPendingBilling(ctx, geminiBatchSettledPrefix+batchKey)
	if err != nil || len(raw) == 0 {
		return false, err
	}
	return true, nil
}

func (s *GeminiMessagesCompatService) ScheduleGeminiBatchPending(ctx context.Context, batchKey string, nextCheckAtUnixMilli int64) {
	if s == nil || s.cache == nil || batchKey == "" {
		return
	}
	idx, ok := s.cache.(GeminiInteractionPendingIndex)
	if !ok {
		return
	}
	_ = idx.UpsertGeminiInteractionPending(ctx, geminiBatchPendingPrefix+batchKey, nextCheckAtUnixMilli, geminiBatchPendingTTL)
}

func (s *GeminiMessagesCompatService) RemoveGeminiBatchPending(ctx context.Context, batchKey string) {
	if s == nil || s.cache == nil || batchKey == "" {
		return
	}
	idx, ok := s.cache.(GeminiInteractionPendingIndex)
	if !ok {
		return
	}
	_ = idx.RemoveGeminiInteractionPending(ctx, geminiBatchPendingPrefix+batchKey)
}

// BuildGeminiBatchNextCheckAt computes stepped polling backoff for asynchronous batch reconciliation.
func BuildGeminiBatchNextCheckAt(now time.Time, createdAtUnix int64) int64 {
	age := now.Sub(time.Unix(createdAtUnix, 0))
	delay := 30 * time.Second
	switch {
	case age > 2*time.Hour:
		delay = 15 * time.Minute
	case age > 15*time.Minute:
		delay = 5 * time.Minute
	case age > 2*time.Minute:
		delay = time.Minute
	}
	return now.Add(delay).UnixMilli()
}

// RewriteGeminiExtBodyModel updates request body model fields when upstream model mapping changed.
func RewriteGeminiExtBodyModel(body []byte, mappedModel string) []byte {
	mapped := strings.TrimSpace(mappedModel)
	if mapped == "" || len(body) == 0 || !gjson.ValidBytes(body) {
		return body
	}
	updated := body
	for _, key := range []string{
		"model",
		"agent_config.model",
		"agentConfig.model",
		"config.model",
		"interaction.model",
		"interaction.agent_config.model",
		"interaction.agentConfig.model",
		"bidi_generate_content_setup.model",
		"bidiGenerateContentSetup.model",
	} {
		cur := gjson.GetBytes(updated, key)
		if !cur.Exists() {
			continue
		}
		targetVal := mapped
		if strings.HasPrefix(strings.TrimSpace(cur.String()), "models/") && !strings.HasPrefix(targetVal, "models/") {
			targetVal = "models/" + targetVal
		}
		if next, err := sjson.SetBytes(updated, key, targetVal); err == nil {
			updated = next
		}
	}
	return updated
}

// ForwardGeminiAIStudioExtensionJSON forwards JSON CRUD/action requests to AI Studio and binds created resource IDs.
func (s *GeminiMessagesCompatService) ForwardGeminiAIStudioExtensionJSON(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	spec GeminiAIStudioExtJSONSpec,
) (*ForwardResult, []byte, error) {
	if s == nil || account == nil {
		return nil, nil, errors.New("gemini compat service or account is nil")
	}
	body := spec.Body
	if len(body) > 0 && spec.MappedModel != "" && spec.MappedModel != spec.OriginalModel {
		body = RewriteGeminiExtBodyModel(body, spec.MappedModel)
	}
	var capturedResp []byte
	res, _, _, err := s.forwardGeminiAIStudioEndpoint(ctx, c, account, geminiAIStudioForwardSpec{
		Method:                spec.Method,
		Path:                  spec.Path,
		Query:                 SanitizeGeminiExtQuery(spec.Query),
		Body:                  body,
		Stream:                false,
		OriginalModel:         spec.OriginalModel,
		MappedModel:           spec.MappedModel,
		TransformResponseJSON: spec.TransformResponseJSON,
		OnResponseJSON: func(respBody []byte) {
			capturedResp = respBody
			s.BindGeminiExtResourcesFromJSON(ctx, respBody, spec.BindKind, account.ID, spec.UserID)
			if spec.OnJSON != nil {
				spec.OnJSON(respBody)
			}
		},
	})
	return res, capturedResp, err
}

// ForwardGeminiBatchEnqueue forwards batchCreate / asyncBatchEmbed requests and extracts returned usage if any.
func (s *GeminiMessagesCompatService) ForwardGeminiBatchEnqueue(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	originalModel, mappedModel, action string,
	query url.Values,
	body []byte,
	userID int64,
) (*ForwardResult, []byte, error) {
	if mappedModel == "" {
		mappedModel = originalModel
	}
	endpointPath := "/v1beta/models/" + mappedModel + ":" + action
	return s.ForwardGeminiAIStudioExtensionJSON(ctx, c, account, GeminiAIStudioExtJSONSpec{
		Method:        http.MethodPost,
		Path:          endpointPath,
		Query:         query,
		Body:          body,
		OriginalModel: originalModel,
		MappedModel:   mappedModel,
		UserID:        userID,
		BindKind:      "batch",
	})
}

// FetchGeminiBatchSnapshot fetches the latest upstream batch/operation state JSON for background settlement.
func (s *GeminiMessagesCompatService) FetchGeminiBatchSnapshot(ctx context.Context, account *Account, batchName string) ([]byte, int, error) {
	if s == nil || s.httpUpstream == nil || account == nil {
		return nil, 0, errors.New("gemini batch fetch unavailable")
	}
	trimmed := strings.TrimSpace(strings.TrimPrefix(batchName, "/"))
	var endpointPath string
	switch {
	case strings.HasPrefix(trimmed, "batches/"):
		endpointPath = "/v1beta/" + trimmed
	case strings.HasPrefix(trimmed, "models/"):
		endpointPath = "/v1beta/" + trimmed
	default:
		endpointPath = "/v1beta/batches/" + trimmed
	}
	targetURL, err := s.buildGeminiAIStudioExtURL(account, endpointPath, nil)
	if err != nil {
		return nil, 0, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, targetURL, nil)
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Accept", "application/json")
	if err := s.setGeminiAIStudioAuthHeaders(ctx, account, req.Header); err != nil {
		return nil, 0, err
	}
	resp, err := s.httpUpstream.Do(req, resolveAccountProxyURL(account), account.ID, account.Concurrency)
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = resp.Body.Close() }()
	body, readErr := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if readErr != nil {
		return nil, resp.StatusCode, readErr
	}
	return body, resp.StatusCode, nil
}

// FetchGeminiFileDownload opens a streaming download response from upstream AI Studio Files API.
func (s *GeminiMessagesCompatService) FetchGeminiFileDownload(ctx context.Context, account *Account, fileName string) (*http.Response, error) {
	if s == nil || s.httpUpstream == nil || account == nil {
		return nil, errors.New("gemini file download unavailable")
	}
	fileID := strings.TrimPrefix(strings.TrimSpace(fileName), "files/")
	if !IsSafeGeminiExtID(fileID) {
		return nil, errors.New("invalid batch responsesFile name")
	}
	q := url.Values{}
	q.Set("alt", "media")
	targetURL, err := s.buildGeminiAIStudioExtURL(account, "/download/v1beta/files/"+fileID+":download", q)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, targetURL, nil)
	if err != nil {
		return nil, err
	}
	if err := s.setGeminiAIStudioAuthHeaders(ctx, account, req.Header); err != nil {
		return nil, err
	}
	return s.httpUpstream.Do(req, resolveAccountProxyURL(account), account.ID, account.Concurrency)
}

func addGeminiClaudeUsage(dst *ClaudeUsage, delta *ClaudeUsage) bool {
	if delta == nil {
		return false
	}
	totalDelta := delta.InputTokens + delta.OutputTokens + delta.CacheReadInputTokens + delta.ImageOutputTokens
	if totalDelta <= 0 {
		return false
	}
	dst.InputTokens += delta.InputTokens
	dst.OutputTokens += delta.OutputTokens
	dst.CacheReadInputTokens += delta.CacheReadInputTokens
	dst.ImageOutputTokens += delta.ImageOutputTokens
	return true
}

func extractBatchInlineUsage(data []byte) (*ClaudeUsage, bool) {
	var total ClaudeUsage
	found := false
	for _, listPath := range []string{
		"output.inlinedResponses.inlinedResponses",
		"response.inlinedResponses.inlinedResponses",
		"output.inlinedEmbedContentResponses.inlinedResponses",
		"response.inlinedEmbedContentResponses.inlinedResponses",
	} {
		arr := gjson.GetBytes(data, listPath)
		if !arr.Exists() || !arr.IsArray() {
			continue
		}
		arr.ForEach(func(_, item gjson.Result) bool {
			raw := []byte(item.Raw)
			if respField := item.Get("response"); respField.Exists() {
				if u := extractGeminiUsage([]byte(respField.Raw)); addGeminiClaudeUsage(&total, u) {
					found = true
					return true
				}
			}
			if u := extractGeminiUsage(raw); addGeminiClaudeUsage(&total, u) {
				found = true
			}
			return true
		})
	}
	if !found {
		if u := extractGeminiUsage(data); addGeminiClaudeUsage(&total, u) {
			found = true
		}
	}
	if !found {
		return nil, false
	}
	return &total, true
}

func parseGeminiBatchState(data []byte) (state string, terminal bool, succeeded bool) {
	if len(data) == 0 || !gjson.ValidBytes(data) {
		return "", false, false
	}
	state = strings.TrimSpace(gjson.GetBytes(data, "state").String())
	if state == "" {
		state = strings.TrimSpace(gjson.GetBytes(data, "metadata.state").String())
	}
	switch state {
	case "BATCH_STATE_SUCCEEDED":
		return state, true, true
	case "BATCH_STATE_FAILED", "BATCH_STATE_CANCELLED", "BATCH_STATE_EXPIRED":
		return state, true, false
	}
	if gjson.GetBytes(data, "done").Bool() {
		if gjson.GetBytes(data, "error").Exists() {
			return state, true, false
		}
		return state, true, true
	}
	return state, false, false
}

// ExtractGeminiBatchUsage parses batch status JSON and computes total token usage (streaming responsesFile when present).
func (s *GeminiMessagesCompatService) ExtractGeminiBatchUsage(
	ctx context.Context,
	account *Account,
	userID int64,
	batchJSON []byte,
) (usage *ClaudeUsage, state string, terminal bool, err error) {
	state, terminal, succeeded := parseGeminiBatchState(batchJSON)
	if !terminal || !succeeded {
		return nil, state, terminal, nil
	}
	if inlineUsage, ok := extractBatchInlineUsage(batchJSON); ok {
		return inlineUsage, state, true, nil
	}
	respFile := firstNonEmpty(
		strings.TrimSpace(gjson.GetBytes(batchJSON, "output.responsesFile").String()),
		strings.TrimSpace(gjson.GetBytes(batchJSON, "response.responsesFile").String()),
	)
	if respFile == "" || !strings.HasPrefix(respFile, "files/") || account == nil {
		return nil, state, true, nil
	}
	s.bindGeminiResourceByName(ctx, respFile, account.ID, userID)
	dlResp, dlErr := s.FetchGeminiFileDownload(ctx, account, respFile)
	if dlErr != nil {
		return nil, state, true, dlErr
	}
	defer func() { _ = dlResp.Body.Close() }()
	if dlResp.StatusCode >= 400 {
		return nil, state, true, fmt.Errorf("download batch responsesFile status %d", dlResp.StatusCode)
	}
	var total ClaudeUsage
	found := false
	scanner := bufio.NewScanner(dlResp.Body)
	scanner.Buffer(make([]byte, 64*1024), geminiBatchJSONLScanMax)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		if respNode := gjson.GetBytes(line, "response"); respNode.Exists() {
			if u := extractGeminiUsage([]byte(respNode.Raw)); addGeminiClaudeUsage(&total, u) {
				found = true
				continue
			}
		}
		if u := extractGeminiUsage(line); addGeminiClaudeUsage(&total, u) {
			found = true
		}
	}
	if scanErr := scanner.Err(); scanErr != nil {
		return nil, state, true, scanErr
	}
	if !found {
		return nil, state, true, nil
	}
	return &total, state, true, nil
}

func copyGeminiExtRequestHeaders(dst, src http.Header) {
	for _, h := range []string{
		"Content-Type",
		"Accept",
		"Range",
		"If-None-Match",
		"If-Modified-Since",
		"X-Goog-Upload-Protocol",
		"X-Goog-Upload-Command",
		"X-Goog-Upload-Offset",
		"X-Goog-Upload-Header-Content-Length",
		"X-Goog-Upload-Header-Content-Type",
		"X-Goog-Upload-Content-Length",
		"X-Goog-Upload-Content-Type",
	} {
		for _, v := range src.Values(h) {
			dst.Add(h, v)
		}
	}
}

func buildGatewayPublicUploadURL(c *gin.Context, uploadID string) string {
	scheme := "http"
	if c.Request.TLS != nil || strings.EqualFold(strings.TrimSpace(c.GetHeader("X-Forwarded-Proto")), "https") {
		scheme = "https"
	}
	host := strings.TrimSpace(c.Request.Host)
	q := "upload_id=" + url.QueryEscape(uploadID) + "&upload_protocol=resumable"
	if host == "" || strings.ContainsAny(host, " \t\r\n/\\@") {
		return c.Request.URL.Path + "?" + q
	}
	u := &url.URL{
		Scheme:   scheme,
		Host:     host,
		Path:     path.Clean(c.Request.URL.Path),
		RawQuery: q,
	}
	return u.String()
}

// ForwardGeminiAIStudioStream streams binary uploads (/upload/v1beta) and media downloads (/download/v1beta)
// without buffering 2GB payloads in memory, while pinning resumable upload sessions and account ownership.
func (s *GeminiMessagesCompatService) ForwardGeminiAIStudioStream(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	spec GeminiAIStudioStreamSpec,
) error {
	if s == nil || s.httpUpstream == nil || account == nil {
		return errors.New("gemini stream service or account unavailable")
	}

	var targetURL string
	var err error
	if spec.UploadSession != nil {
		targetURL, err = s.validateUpstreamBaseURL(spec.UploadSession.UpstreamURL)
	} else {
		targetURL, err = s.buildGeminiAIStudioExtURL(account, spec.Path, spec.Query)
	}
	if err != nil {
		return s.writeGoogleError(c, http.StatusBadGateway, "Invalid upstream target URL")
	}

	// 判断请求体是否可重放：GET 或已知小于 1MiB 的小载荷支持同号重试/切号，大于 1MiB 或分块流式直通。
	var bufferedBody []byte
	replayable := spec.Method == http.MethodGet || spec.Method == http.MethodDelete || spec.Method == http.MethodHead
	if !replayable && c.Request.Body != nil && c.Request.ContentLength >= 0 && c.Request.ContentLength <= geminiStreamSmallRetryCap {
		bufferedBody, err = io.ReadAll(io.LimitReader(c.Request.Body, geminiStreamSmallRetryCap+1))
		if err != nil {
			return s.writeGoogleError(c, http.StatusBadRequest, "Failed to read request body")
		}
		if len(bufferedBody) <= geminiStreamSmallRetryCap {
			replayable = true
		}
	}

	proxyURL := resolveAccountProxyURL(account)
	maxAttempts := 1
	if replayable && spec.UploadSession == nil {
		maxAttempts = s.geminiMaxAttemptsPerAccount()
		if maxAttempts < 1 {
			maxAttempts = 1
		}
	}

	resetReplayableBody := func() {
		if replayable && len(bufferedBody) > 0 && c != nil && c.Request != nil {
			c.Request.Body = io.NopCloser(bytes.NewReader(bufferedBody))
		}
	}

	var resp *http.Response
	var checkedPolicy *ErrorPolicyResult
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		var bodyReader io.Reader
		if replayable {
			if len(bufferedBody) > 0 {
				bodyReader = bytes.NewReader(bufferedBody)
			}
		} else if c.Request.Body != nil {
			bodyReader = c.Request.Body
		}
		req, reqErr := http.NewRequestWithContext(ctx, spec.Method, targetURL, bodyReader)
		if reqErr != nil {
			return s.writeGoogleError(c, http.StatusBadGateway, "Failed to construct upstream request")
		}
		if replayable {
			req.ContentLength = int64(len(bufferedBody))
			if len(bufferedBody) > 0 {
				bodyCopy := bufferedBody
				req.GetBody = func() (io.ReadCloser, error) {
					return io.NopCloser(bytes.NewReader(bodyCopy)), nil
				}
			}
		} else if c.Request.ContentLength >= 0 {
			req.ContentLength = c.Request.ContentLength
		}

		copyGeminiExtRequestHeaders(req.Header, c.Request.Header)
		if authErr := s.setGeminiAIStudioAuthHeaders(ctx, account, req.Header); authErr != nil {
			return authErr
		}

		resp, err = s.httpUpstream.Do(req, proxyURL, account.ID, account.Concurrency)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if replayable && attempt < maxAttempts {
				sleepGeminiBackoff(attempt)
				continue
			}
			if replayable {
				resetReplayableBody()
				return &UpstreamFailoverError{StatusCode: http.StatusBadGateway}
			}
			return s.writeGoogleError(c, http.StatusBadGateway, "Upstream transport error")
		}

		checked, rebuilt := s.checkErrorPolicyInLoop(ctx, account, resp, spec.MappedModel)
		resp = rebuilt
		if checked != nil && *checked != ErrorPolicyNone {
			checkedPolicy = checked
			break
		}

		if resp.StatusCode >= 400 && replayable && attempt < maxAttempts && s.shouldRetryGeminiUpstreamError(account, resp.StatusCode) {
			_ = resp.Body.Close()
			logger.LegacyPrintf("service.gemini_ext_endpoints", "Gemini account %d stream status %d retry %d/%d", account.ID, resp.StatusCode, attempt, maxAttempts)
			sleepGeminiBackoff(attempt)
			continue
		}
		break
	}
	if resp == nil {
		return s.writeGoogleError(c, http.StatusBadGateway, "Empty upstream response")
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= 400 {
		respBody := s.readUpstreamErrorBody(resp)
		upstreamReqID := resp.Header.Get("x-request-id")
		if upstreamReqID == "" {
			upstreamReqID = resp.Header.Get("x-goog-request-id")
		}
		if s.rateLimitService != nil {
			policy := s.finalErrorPolicy(ctx, account, resp.StatusCode, respBody, spec.MappedModel, checkedPolicy)
			switch policy {
			case ErrorPolicySkipped:
				if replayable {
					resetReplayableBody()
					if failoverErr := s.skippedErrorPolicyFailoverError(c, account, resp.StatusCode, respBody, upstreamReqID); failoverErr != nil {
						return failoverErr
					}
				}
				if account.IsCustomErrorCodesEnabled() {
					return s.writeGeminiCustomCodeSkippedError(c, account, resp.StatusCode, upstreamReqID, respBody, func() {
						_ = s.writeGoogleError(c, http.StatusInternalServerError, geminiCustomCodeSkippedClientMessage)
					})
				}
				return s.writeGeminiMappedError(c, account, resp.StatusCode, upstreamReqID, respBody)
			case ErrorPolicyMatched, ErrorPolicyTempUnscheduled:
				if policy == ErrorPolicyMatched {
					s.handleGeminiUpstreamError(ctx, account, resp.StatusCode, resp.Header, respBody)
				}
				if replayable {
					resetReplayableBody()
					return &UpstreamFailoverError{StatusCode: resp.StatusCode, ResponseBody: respBody, ResponseHeaders: resp.Header.Clone()}
				}
				return s.writeGeminiMappedError(c, account, resp.StatusCode, upstreamReqID, respBody)
			}
		}

		s.handleGeminiUpstreamError(ctx, account, resp.StatusCode, resp.Header, respBody)
		if replayable && s.shouldFailoverGeminiUpstreamError(resp.StatusCode) {
			resetReplayableBody()
			return &UpstreamFailoverError{StatusCode: resp.StatusCode, ResponseBody: respBody, ResponseHeaders: resp.Header.Clone()}
		}
		return s.writeGeminiMappedError(c, account, resp.StatusCode, upstreamReqID, respBody)
	}

	rawUploadURL := strings.TrimSpace(resp.Header.Get("X-Goog-Upload-URL"))
	var rewrittenUploadURL string
	if rawUploadURL != "" {
		validatedUploadURL, valErr := s.validateUpstreamBaseURL(rawUploadURL)
		if valErr != nil {
			return s.writeGoogleError(c, http.StatusBadGateway, "Upstream returned unsafe upload URL")
		}
		newUploadID := strings.ReplaceAll(uuid.NewString(), "-", "")
		if saveErr := s.SaveGeminiUploadSession(ctx, newUploadID, GeminiUploadSession{
			AccountID:     account.ID,
			UserID:        spec.UserID,
			UpstreamURL:   validatedUploadURL,
			TargetPath:    spec.Path,
			OriginalModel: spec.OriginalModel,
		}); saveErr != nil {
			return s.writeGoogleError(c, http.StatusInternalServerError, "Failed to persist upload session")
		}
		rewrittenUploadURL = buildGatewayPublicUploadURL(c, newUploadID)
	}

	responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	for _, h := range geminiExtPassThroughUploadHeaders {
		if val := resp.Header.Get(h); val != "" {
			c.Writer.Header().Set(h, val)
		}
	}
	if rewrittenUploadURL != "" {
		c.Writer.Header().Set("X-Goog-Upload-URL", rewrittenUploadURL)
	} else {
		c.Writer.Header().Del("X-Goog-Upload-URL")
	}
	if spec.IsDownload && resp.ContentLength >= 0 {
		c.Writer.Header().Set("Content-Length", strconv.FormatInt(resp.ContentLength, 10))
	}

	contentType := strings.ToLower(resp.Header.Get("Content-Type"))
	if !spec.IsDownload && strings.Contains(contentType, "application/json") {
		respBytes, readErr := io.ReadAll(io.LimitReader(resp.Body, geminiStreamJSONRespCap+1))
		if readErr != nil {
			return s.writeGoogleError(c, http.StatusBadGateway, "Failed to read upstream JSON response")
		}
		if len(respBytes) > geminiStreamJSONRespCap {
			// Stream the complete response if it is too large to inspect.
			c.Status(resp.StatusCode)
			_, _ = io.Copy(c.Writer, io.MultiReader(bytes.NewReader(respBytes), resp.Body))
			return nil
		}
		s.BindGeminiExtResourcesFromJSON(ctx, respBytes, spec.BindKind, account.ID, spec.UserID)
		c.Status(resp.StatusCode)
		if len(respBytes) > 0 {
			_, _ = c.Writer.Write(respBytes)
		}
		return nil
	}

	c.Status(resp.StatusCode)
	buf := make([]byte, 64*1024)
	_, _ = io.CopyBuffer(c.Writer, resp.Body, buf)
	return nil
}
