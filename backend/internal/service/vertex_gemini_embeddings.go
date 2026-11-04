package service

import (
	"encoding/json"
	"errors"
	"math"
	"strings"

	"github.com/tidwall/gjson"
)

// Gemini's synchronous batch endpoint accepts at most 100 embedding requests.
const maxVertexGeminiEmbeddingBatch = 100

// vertexGeminiEmbeddingRequests validates every item before starting any paid
// upstream calls. Vertex takes the model from the URL, not from the body.
func vertexGeminiEmbeddingRequests(body []byte, action string, models ...string) ([][]byte, error) {
	items := []json.RawMessage{body}
	if action == "batchEmbedContents" {
		var batch struct {
			Requests []json.RawMessage `json:"requests"`
		}
		if err := json.Unmarshal(body, &batch); err != nil || len(batch.Requests) == 0 {
			return nil, errors.New("embedding batch must contain requests")
		}
		if len(batch.Requests) > maxVertexGeminiEmbeddingBatch {
			return nil, errors.New("embedding batch may contain at most 100 requests")
		}
		items = batch.Requests
	}

	requests := make([][]byte, 0, len(items))
	for _, item := range items {
		var request map[string]json.RawMessage
		if err := json.Unmarshal(item, &request); err != nil || request == nil {
			return nil, errors.New("invalid embedding request")
		}
		if model, exists := request["model"]; exists {
			var value string
			if err := json.Unmarshal(model, &value); err != nil || strings.TrimSpace(value) == "" {
				return nil, errors.New("invalid embedding request model")
			}
			value = strings.TrimPrefix(strings.TrimSpace(value), "models/")
			matched := false
			for _, candidate := range models {
				if value == strings.TrimPrefix(strings.TrimSpace(candidate), "models/") && value != "" {
					matched = true
					break
				}
			}
			if !matched {
				return nil, errors.New("embedding request model does not match the requested model")
			}
		}
		var content struct {
			Parts []json.RawMessage `json:"parts"`
		}
		if err := json.Unmarshal(request["content"], &content); err != nil || len(content.Parts) == 0 {
			return nil, errors.New("embedding request content must contain parts")
		}
		for _, part := range content.Parts {
			var object map[string]json.RawMessage
			if err := json.Unmarshal(part, &object); err != nil || len(object) == 0 {
				return nil, errors.New("invalid embedding content part")
			}
		}
		delete(request, "model")
		encoded, err := json.Marshal(request)
		if err != nil {
			return nil, errors.New("invalid embedding request")
		}
		requests = append(requests, encoded)
	}
	return requests, nil
}

type vertexGeminiEmbeddingTokenDetail struct {
	Modality   string `json:"modality"`
	TokenCount int    `json:"tokenCount"`
}

// vertexGeminiEmbeddingBatch turns ordered Vertex single-item responses into
// the Gemini batch response expected by both native and OpenAI clients.
type vertexGeminiEmbeddingBatch struct {
	requests    [][]byte
	embeddings  []json.RawMessage
	inputTokens int
	details     []vertexGeminiEmbeddingTokenDetail
	usageSeen   bool
	readBytes   int64
}

func (b *vertexGeminiEmbeddingBatch) addResponse(body []byte, maxBytes int64) error {
	b.readBytes += int64(len(body))
	if b.readBytes > maxBytes {
		return ErrUpstreamResponseBodyTooLarge
	}
	var response struct {
		Embedding json.RawMessage `json:"embedding"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return errors.New("invalid upstream embedding response")
	}
	var vector geminiEmbeddingVector
	if err := json.Unmarshal(response.Embedding, &vector); err != nil || len(vector.Values) == 0 {
		return errors.New("invalid upstream embedding response")
	}
	for _, value := range vector.Values {
		if value == nil || math.IsInf(float64(float32(*value)), 0) {
			return errors.New("invalid upstream embedding response")
		}
	}
	b.embeddings = append(b.embeddings, response.Embedding)
	usage := gjson.GetBytes(body, "usageMetadata")
	if usage.Exists() {
		b.usageSeen = true
	}
	b.inputTokens += firstPositiveGJSONInt(usage.Get("promptTokenCount"), gjson.GetBytes(body, "tokenCount"), usage.Get("totalTokenCount"))
	details := usage.Get("promptTokensDetails")
	if !details.Exists() {
		details = usage.Get("promptTokenDetails")
	}
	for _, detail := range details.Array() {
		modality := detail.Get("modality").String()
		count := int(detail.Get("tokenCount").Int())
		if modality == "" || count <= 0 {
			continue
		}
		found := false
		for i := range b.details {
			if b.details[i].Modality == modality {
				b.details[i].TokenCount += count
				found = true
				break
			}
		}
		if !found {
			b.details = append(b.details, vertexGeminiEmbeddingTokenDetail{Modality: modality, TokenCount: count})
		}
	}
	return nil
}

func (b *vertexGeminiEmbeddingBatch) response() ([]byte, error) {
	result := map[string]any{"embeddings": b.embeddings}
	if b.usageSeen || b.inputTokens > 0 {
		usage := map[string]any{"promptTokenCount": b.inputTokens}
		if len(b.details) > 0 {
			usage["promptTokenDetails"] = b.details
		}
		result["usageMetadata"] = usage
	}
	return json.Marshal(result)
}
