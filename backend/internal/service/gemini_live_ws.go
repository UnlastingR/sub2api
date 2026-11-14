package service

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"strings"
	"sync"

	coderws "github.com/coder/websocket"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	geminiLiveBidiPath            = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
	geminiLiveBidiConstrainedPath = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained"
)

// GeminiLiveWSConn abstracts raw WebSocket frame transport for Gemini Live bidirectional proxy.
type GeminiLiveWSConn interface {
	ReadFrame(ctx context.Context) (coderws.MessageType, []byte, error)
	WriteFrame(ctx context.Context, msgType coderws.MessageType, payload []byte) error
	Close() error
	CloseNow() error
}

func (s *GeminiMessagesCompatService) buildGeminiLiveWSURL(
	account *Account,
	constrained bool,
	rawQuery url.Values,
	accessTokenOverride string,
) (string, error) {
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
	switch strings.ToLower(u.Scheme) {
	case "https", "wss":
		u.Scheme = "wss"
	case "http", "ws":
		u.Scheme = "ws"
	default:
		return "", errors.New("unsupported upstream scheme for websocket")
	}

	trimmedBasePath := strings.TrimRight(u.Path, "/")
	for _, ver := range []string{"/v1beta", "/v1alpha", "/v1"} {
		if strings.HasSuffix(trimmedBasePath, ver) {
			trimmedBasePath = strings.TrimSuffix(trimmedBasePath, ver)
			break
		}
	}
	wsPath := geminiLiveBidiPath
	if constrained {
		wsPath = geminiLiveBidiConstrainedPath
	}
	u.Path = trimmedBasePath + wsPath

	q := SanitizeGeminiExtQuery(rawQuery)
	if tok := strings.TrimSpace(accessTokenOverride); tok != "" {
		if q == nil {
			q = make(url.Values)
		}
		q.Set("access_token", tok)
	}
	if len(q) > 0 {
		u.RawQuery = q.Encode()
	} else {
		u.RawQuery = ""
	}
	return u.String(), nil
}

// DialGeminiLiveWS establishes an upstream WebSocket connection to Gemini Live API.
func (s *GeminiMessagesCompatService) DialGeminiLiveWS(
	ctx context.Context,
	account *Account,
	constrained bool,
	rawQuery url.Values,
	accessTokenOverride string,
) (GeminiLiveWSConn, int, http.Header, error) {
	if s == nil || account == nil {
		return nil, http.StatusBadGateway, nil, errors.New("gemini live service or account unavailable")
	}
	useConstrainedUpstream := constrained && strings.TrimSpace(accessTokenOverride) != ""
	wsURL, err := s.buildGeminiLiveWSURL(account, useConstrainedUpstream, rawQuery, accessTokenOverride)
	if err != nil {
		return nil, http.StatusBadGateway, nil, err
	}

	headers := make(http.Header)
	if !useConstrainedUpstream {
		if authErr := s.setGeminiAIStudioAuthHeaders(ctx, account, headers); authErr != nil {
			return nil, http.StatusBadGateway, nil, authErr
		}
	}

	dialer := newDefaultOpenAIWSClientDialer()
	conn, statusCode, respHeader, dialErr := dialer.Dial(ctx, wsURL, headers, resolveAccountProxyURL(account))
	if dialErr != nil {
		return nil, statusCode, respHeader, dialErr
	}
	frameConn, ok := conn.(GeminiLiveWSConn)
	if !ok {
		_ = conn.Close()
		return nil, http.StatusBadGateway, respHeader, errors.New("upstream websocket conn does not support raw frames")
	}
	return frameConn, statusCode, respHeader, nil
}

// ExtractGeminiLiveUsage parses Live API usageMetadata (which reports responseTokenCount + toolUsePromptTokenCount).
func ExtractGeminiLiveUsage(data []byte) *ClaudeUsage {
	if len(data) == 0 || !gjson.ValidBytes(data) {
		return nil
	}
	usage := gjson.GetBytes(data, "usageMetadata")
	if !usage.Exists() {
		usage = gjson.GetBytes(data, "serverContent.usageMetadata")
	}
	if !usage.Exists() {
		return nil
	}

	prompt := int(usage.Get("promptTokenCount").Int())
	toolPrompt := int(usage.Get("toolUsePromptTokenCount").Int())
	cached := int(usage.Get("cachedContentTokenCount").Int())
	thoughts := int(usage.Get("thoughtsTokenCount").Int())
	responseTokens := int(usage.Get("responseTokenCount").Int())
	if responseTokens == 0 {
		responseTokens = int(usage.Get("candidatesTokenCount").Int())
	}

	audioInputTokens := 0
	if promptDetails := usage.Get("promptTokensDetails"); promptDetails.Exists() {
		promptDetails.ForEach(func(_, item gjson.Result) bool {
			if strings.EqualFold(item.Get("modality").String(), "AUDIO") {
				audioInputTokens += int(item.Get("tokenCount").Int())
			}
			return true
		})
	}
	imageTokens := 0
	audioOutputTokens := 0
	for _, detailField := range []string{"responseTokensDetails", "candidatesTokensDetails"} {
		details := usage.Get(detailField)
		if !details.Exists() {
			continue
		}
		details.ForEach(func(_, item gjson.Result) bool {
			switch strings.ToUpper(strings.TrimSpace(item.Get("modality").String())) {
			case "IMAGE":
				imageTokens += int(item.Get("tokenCount").Int())
			case "AUDIO":
				audioOutputTokens += int(item.Get("tokenCount").Int())
			}
			return true
		})
	}

	rawInput := prompt + toolPrompt
	return &ClaudeUsage{
		InputTokens:          max(rawInput-cached, 0),
		OutputTokens:         responseTokens + thoughts,
		CacheReadInputTokens: cached,
		ImageOutputTokens:    imageTokens,
		AudioInputTokens:     audioInputTokens,
		AudioOutputTokens:    audioOutputTokens,
	}
}

// GeminiLiveUsageTracker accumulates multi-turn Live API session usage without double-counting mid-turn snapshots.
type GeminiLiveUsageTracker struct {
	mu          sync.Mutex
	committed   ClaudeUsage
	currentTurn ClaudeUsage
	hasTurnData bool
}

// ObserveServerFrame inspects a server WebSocket frame, updating peak per-turn usage and committing completed turns.
func (t *GeminiLiveUsageTracker) ObserveServerFrame(payload []byte) {
	if t == nil || len(payload) == 0 || !gjson.ValidBytes(payload) {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	if u := ExtractGeminiLiveUsage(payload); u != nil {
		t.currentTurn.InputTokens = max(t.currentTurn.InputTokens, u.InputTokens)
		t.currentTurn.OutputTokens = max(t.currentTurn.OutputTokens, u.OutputTokens)
		t.currentTurn.CacheReadInputTokens = max(t.currentTurn.CacheReadInputTokens, u.CacheReadInputTokens)
		t.currentTurn.ImageOutputTokens = max(t.currentTurn.ImageOutputTokens, u.ImageOutputTokens)
		t.currentTurn.AudioInputTokens = max(t.currentTurn.AudioInputTokens, u.AudioInputTokens)
		t.currentTurn.AudioOutputTokens = max(t.currentTurn.AudioOutputTokens, u.AudioOutputTokens)
		t.hasTurnData = true
	}

	sc := gjson.GetBytes(payload, "serverContent")
	if sc.Exists() && (sc.Get("turnComplete").Bool() || sc.Get("interrupted").Bool()) {
		t.commitCurrentTurnLocked()
	}
}

func (t *GeminiLiveUsageTracker) commitCurrentTurnLocked() {
	if !t.hasTurnData {
		return
	}
	t.committed.InputTokens += t.currentTurn.InputTokens
	t.committed.OutputTokens += t.currentTurn.OutputTokens
	t.committed.CacheReadInputTokens += t.currentTurn.CacheReadInputTokens
	t.committed.ImageOutputTokens += t.currentTurn.ImageOutputTokens
	t.committed.AudioInputTokens += t.currentTurn.AudioInputTokens
	t.committed.AudioOutputTokens += t.currentTurn.AudioOutputTokens
	t.currentTurn = ClaudeUsage{}
	t.hasTurnData = false
}

// Total commits any open turn snapshot and returns cumulative session usage.
func (t *GeminiLiveUsageTracker) Total() ClaudeUsage {
	if t == nil {
		return ClaudeUsage{}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.commitCurrentTurnLocked()
	return t.committed
}

// ParseGeminiLiveSetupModel extracts the requested model name from a Live API setup frame.
func ParseGeminiLiveSetupModel(frame []byte) string {
	if len(frame) == 0 || !gjson.ValidBytes(frame) {
		return ""
	}
	raw := strings.TrimSpace(gjson.GetBytes(frame, "setup.model").String())
	return strings.TrimPrefix(raw, "models/")
}

// RewriteGeminiLiveSetupModel rewrites setup.model to the upstream mapped model name while preserving payload fields.
func RewriteGeminiLiveSetupModel(frame []byte, mappedModel string) ([]byte, bool) {
	mapped := strings.TrimPrefix(strings.TrimSpace(mappedModel), "models/")
	if mapped == "" || len(frame) == 0 || !gjson.ValidBytes(frame) {
		return frame, false
	}
	setup := gjson.GetBytes(frame, "setup.model")
	if !setup.Exists() {
		return frame, false
	}
	updated, err := sjson.SetBytes(frame, "setup.model", "models/"+mapped)
	if err != nil {
		return frame, false
	}
	return updated, true
}

// ExtractGeminiLiveAuditText extracts client text messages from Live WS frames for prompt security auditing.
func ExtractGeminiLiveAuditText(frame []byte) string {
	if len(frame) == 0 || !gjson.ValidBytes(frame) {
		return ""
	}
	var parts []string
	collectPartText := func(arr gjson.Result) {
		if !arr.Exists() || !arr.IsArray() {
			return
		}
		arr.ForEach(func(_, item gjson.Result) bool {
			if txt := strings.TrimSpace(item.Get("text").String()); txt != "" {
				parts = append(parts, txt)
			}
			return true
		})
	}
	collectPartText(gjson.GetBytes(frame, "setup.systemInstruction.parts"))
	if turns := gjson.GetBytes(frame, "clientContent.turns"); turns.Exists() && turns.IsArray() {
		turns.ForEach(func(_, turn gjson.Result) bool {
			collectPartText(turn.Get("parts"))
			return true
		})
	}
	if txt := strings.TrimSpace(gjson.GetBytes(frame, "realtimeInput.text").String()); txt != "" {
		parts = append(parts, txt)
	}
	return strings.Join(parts, "\n")
}

// BuildGeminiLiveAuditJSON wraps extracted Live WS text into a valid Gemini generateContent
// JSON payload so downstream security audit and content moderation parsers inspect it cleanly.
func BuildGeminiLiveAuditJSON(frame []byte) []byte {
	txt := ExtractGeminiLiveAuditText(frame)
	if txt == "" {
		return nil
	}
	payload, err := json.Marshal(map[string]any{
		"contents": []map[string]any{
			{
				"role": "user",
				"parts": []map[string]string{
					{"text": txt},
				},
			},
		},
	})
	if err != nil {
		return nil
	}
	return payload
}
