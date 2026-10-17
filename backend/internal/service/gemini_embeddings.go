package service

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/util/responseheaders"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
)

const geminiEmbeddingResponseOptionsKey = "gemini_openai_embedding_options"

type geminiEmbeddingResponseOptions struct {
	OriginalBody   []byte
	Model          string
	EncodingFormat string
	InputCount     int
}

type geminiOpenAIEmbeddingRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	Dimensions     *int            `json:"dimensions"`
	EncodingFormat string          `json:"encoding_format"`
	Stream         bool            `json:"stream"`
}

func parseGeminiEmbeddingsRequest(body []byte) (*geminiOpenAIEmbeddingRequest, []string, error) {
	var req geminiOpenAIEmbeddingRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, nil, errors.New("invalid embeddings request body")
	}
	req.Model = strings.TrimSpace(req.Model)
	model := strings.TrimPrefix(req.Model, "models/")
	if !IsSafeGeminiModelPathSegment(model) {
		return nil, nil, errors.New("a valid embedding model is required")
	}
	if req.Stream {
		return nil, nil, errors.New("embeddings do not support streaming")
	}
	if req.EncodingFormat == "" {
		req.EncodingFormat = "float"
	}
	if req.EncodingFormat != "float" && req.EncodingFormat != "base64" {
		return nil, nil, errors.New("encoding_format must be float or base64")
	}
	if req.Dimensions != nil && *req.Dimensions <= 0 {
		return nil, nil, errors.New("dimensions must be a positive integer")
	}
	var inputs []string
	var single string
	if json.Unmarshal(req.Input, &single) == nil && string(req.Input) != "null" {
		inputs = []string{single}
	} else if err := json.Unmarshal(req.Input, &inputs); err != nil {
		return nil, nil, errors.New("input must be a string or a non-empty array of strings; token arrays are not supported")
	}
	if len(inputs) == 0 {
		return nil, nil, errors.New("input must be a string or a non-empty array of strings")
	}
	for _, input := range inputs {
		if strings.TrimSpace(input) == "" {
			return nil, nil, errors.New("input strings must not be empty")
		}
	}
	return &req, inputs, nil
}

// PrepareGeminiEmbeddingsRequest adapts only the request representation. The
// handler delegates to GeminiV1BetaModels for the existing native gateway flow.
func PrepareGeminiEmbeddingsRequest(c *gin.Context, body []byte) (string, []byte, error) {
	req, inputs, err := parseGeminiEmbeddingsRequest(body)
	if err != nil {
		return "", nil, err
	}
	model := strings.TrimPrefix(req.Model, "models/")
	requests := make([]map[string]any, len(inputs))
	for i, input := range inputs {
		item := map[string]any{
			"model":   "models/" + model,
			"content": map[string]any{"parts": []map[string]string{{"text": input}}},
		}
		if req.Dimensions != nil {
			item["outputDimensionality"] = *req.Dimensions
		}
		requests[i] = item
	}
	nativeBody, err := json.Marshal(map[string]any{"requests": requests})
	if err != nil {
		return "", nil, err
	}
	c.Set(geminiEmbeddingResponseOptionsKey, &geminiEmbeddingResponseOptions{
		OriginalBody: body, Model: req.Model, EncodingFormat: req.EncodingFormat, InputCount: len(inputs),
	})
	return model, nativeBody, nil
}

func geminiEmbeddingOptionsFromContext(c *gin.Context) *geminiEmbeddingResponseOptions {
	value, _ := c.Get(geminiEmbeddingResponseOptionsKey)
	options, _ := value.(*geminiEmbeddingResponseOptions)
	return options
}

// GeminiEmbeddingsOriginalBody preserves the client representation for the
// existing prompt audit; the native adapter must not bypass that check.
func GeminiEmbeddingsOriginalBody(c *gin.Context) []byte {
	if options := geminiEmbeddingOptionsFromContext(c); options != nil {
		return options.OriginalBody
	}
	return nil
}

// Account-level aliases must be applied to both the URL and each batch item's
// model, otherwise Google rejects an otherwise valid embedding request.
func mapGeminiEmbeddingRequestModels(body []byte, model, action string) ([]byte, error) {
	var root map[string]json.RawMessage
	if err := json.Unmarshal(body, &root); err != nil || root == nil {
		return nil, errors.New("invalid embedding request")
	}
	mapped, err := json.Marshal("models/" + model)
	if err != nil {
		return nil, err
	}
	if action == "embedContent" {
		if _, exists := root["model"]; exists {
			root["model"] = mapped
		}
	} else {
		var requests []map[string]json.RawMessage
		if err := json.Unmarshal(root["requests"], &requests); err != nil {
			return nil, err
		}
		for _, request := range requests {
			if request == nil {
				return nil, errors.New("invalid embedding batch item")
			}
			request["model"] = mapped
		}
		root["requests"], err = json.Marshal(requests)
		if err != nil {
			return nil, err
		}
	}
	return json.Marshal(root)
}

type geminiEmbeddingVector struct {
	Values []*float64 `json:"values"`
}

func (s *GeminiMessagesCompatService) handleGeminiEmbeddingResponse(c *gin.Context, resp *http.Response, body []byte, options *geminiEmbeddingResponseOptions) (*ClaudeUsage, error) {
	var envelope struct {
		Embeddings []geminiEmbeddingVector `json:"embeddings"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, s.writeChatCompletionsError(c, http.StatusBadGateway, "upstream_error", "Invalid upstream embeddings response")
	}
	if len(envelope.Embeddings) != options.InputCount {
		return nil, s.writeChatCompletionsError(c, http.StatusBadGateway, "upstream_error", "Upstream embedding count does not match the request")
	}
	data := make([]map[string]any, len(envelope.Embeddings))
	for i, vector := range envelope.Embeddings {
		if len(vector.Values) == 0 {
			return nil, s.writeChatCompletionsError(c, http.StatusBadGateway, "upstream_error", "Upstream embedding values are empty")
		}
		values := make([]float64, len(vector.Values))
		for j, value := range vector.Values {
			if value == nil || math.IsInf(float64(float32(*value)), 0) {
				return nil, s.writeChatCompletionsError(c, http.StatusBadGateway, "upstream_error", "Upstream embedding values are invalid")
			}
			values[j] = *value
		}
		var embedding any = values
		if options.EncodingFormat == "base64" {
			buf := make([]byte, len(values)*4)
			for j, value := range values {
				binary.LittleEndian.PutUint32(buf[j*4:], math.Float32bits(float32(value)))
			}
			embedding = base64.StdEncoding.EncodeToString(buf)
		}
		data[i] = map[string]any{"object": "embedding", "index": i, "embedding": embedding}
	}
	inputTokens := firstPositiveGJSONInt(gjson.GetBytes(body, "usageMetadata.promptTokenCount"), gjson.GetBytes(body, "tokenCount"), gjson.GetBytes(body, "usageMetadata.totalTokenCount"))
	responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	c.Writer.Header().Del("Content-Length")
	c.Writer.Header().Del("Content-Encoding")
	c.JSON(resp.StatusCode, map[string]any{
		"object": "list", "model": options.Model, "data": data,
		"usage": map[string]int{"prompt_tokens": inputTokens, "total_tokens": inputTokens},
	})
	return &ClaudeUsage{InputTokens: inputTokens}, nil
}
