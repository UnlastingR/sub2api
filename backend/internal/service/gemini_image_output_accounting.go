package service

import (
	"bytes"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	_ "golang.org/x/image/webp"
)

// geminiImageOutputCounterKey 是请求级图片计数器挂在 gin.Context 上的键。
const geminiImageOutputCounterKey = "gemini_image_output_counter"

// geminiImageOutputCounter 记录一次转发里 Gemini 上游真正回吐的图片数量。
//
// 取「单个 payload 内的最大值」而不是累加：Gemini 兼容上游的 SSE 分片可能是
// 累积式的（同一份内容在后续 chunk 里重复整段回来——本文件同目录的
// computeGeminiTextDelta 就是为此存在的），逐 chunk 累加会把同一张图算很多次。
// 计费上宁可少算不可多算，所以用 max 兜底：
//   - 非流式：整份响应体只观测一次，max 即真实张数；
//   - 累积式流：最后一个 chunk 含全部图片，max 仍是真实张数；
//   - 增量式流且多图分散在不同 chunk：会低估到 1，与改动前的模型名启发式同值，
//     不构成回退。
type geminiImageOutputCounter struct {
	count       int
	outputSizes []string
}

// beginGeminiImageOutputObservation 在每次 Forward 开头重置计数器。
// failover 会拿同一个 gin.Context 重跑转发，不重置就会把上一个账号的图数带进来。
func beginGeminiImageOutputObservation(c *gin.Context) *geminiImageOutputCounter {
	if c == nil {
		return nil
	}
	counter := &geminiImageOutputCounter{}
	c.Set(geminiImageOutputCounterKey, counter)
	return counter
}

func geminiImageOutputCounterFromContext(c *gin.Context) *geminiImageOutputCounter {
	if c == nil {
		return nil
	}
	value, ok := c.Get(geminiImageOutputCounterKey)
	if !ok {
		return nil
	}
	counter, _ := value.(*geminiImageOutputCounter)
	return counter
}

// observeGeminiImageOutputs 观测一段上游响应（整份或单个 chunk）里的图片。
// 调用点与 upstreamResponseModelObserver.ObserveGemini 一一对应——那里拿得到
// 解包后的上游响应体，这里需要的是同一份字节。
func observeGeminiImageOutputs(c *gin.Context, payload []byte) {
	counter := geminiImageOutputCounterFromContext(c)
	if counter == nil {
		return
	}
	count, sizes := extractGeminiImageOutputs(payload)
	if count > counter.count {
		counter.count = count
		counter.outputSizes = sizes
	} else if count == counter.count && len(sizes) > len(counter.outputSizes) {
		counter.outputSizes = sizes
	}
}

func observedGeminiImageOutputs(c *gin.Context) int {
	counter := geminiImageOutputCounterFromContext(c)
	if counter == nil {
		return 0
	}
	return counter.count
}

func observedGeminiImageOutputSizes(c *gin.Context) []string {
	counter := geminiImageOutputCounterFromContext(c)
	if counter == nil || len(counter.outputSizes) == 0 {
		return nil
	}
	out := make([]string, len(counter.outputSizes))
	copy(out, counter.outputSizes)
	return out
}

// resolveGeminiImageCount 决定本次请求按几张图计费。
//
// 优先用上游真正返回的图片数：走 GeminiMessagesCompatService 的账号多是
// API Key + 自定义模型映射，客户端请求名和上游模型名都可能是站长自取的别名
// （issue #5358 里的 nana-banana-2），isImageGenerationModel 的白名单必然判不出，
// 于是 ImageCount=0，calculateRecordUsageCost 整条按次计费分支不触发，
// 生图请求全部记 $0。
//
// 只有响应里数不出图时（例如聚合函数丢掉了图片 part）才退回既有的模型名
// 启发式，保证老行为不回退；这里额外也认映射后的上游模型名，与 shouldSkipCodexPlanGatedImageModelCooldown 对
// requestedModel / modelKey 双取的口径一致。
func resolveGeminiImageCount(c *gin.Context, originalModel, mappedModel string) int {
	if observed := observedGeminiImageOutputs(c); observed > 0 {
		return observed
	}
	if isImageGenerationModel(originalModel) || isImageGenerationModel(mappedModel) {
		return 1
	}
	return 0
}

// countGeminiImageOutputs 统计一段 Gemini 响应 JSON 里的图片 part。
// 同时识别 inlineData、fileData 及 SDK/中转使用的 snake_case 字段。
func countGeminiImageOutputs(payload []byte) int {
	count, _ := extractGeminiImageOutputs(payload)
	return count
}

func extractGeminiImageOutputs(payload []byte) (int, []string) {
	if len(payload) == 0 || !gjson.ValidBytes(payload) {
		return 0, nil
	}
	count := 0
	var sizes []string
	gjson.GetBytes(payload, "candidates").ForEach(func(_, candidate gjson.Result) bool {
		candidate.Get("content.parts").ForEach(func(_, part gjson.Result) bool {
			if isImg, dims := geminiPartImageInfo(part); isImg {
				count++
				if dims != "" {
					sizes = append(sizes, dims)
				}
			}
			return true
		})
		return true
	})
	return count, sizes
}

func geminiPartImageInfo(part gjson.Result) (bool, string) {
	for _, key := range []string{"inlineData", "inline_data", "fileData", "file_data"} {
		imagePart := part.Get(key)
		mimeType := imagePart.Get("mimeType")
		if !mimeType.Exists() {
			mimeType = imagePart.Get("mime_type")
		}
		if !isGeminiInlineImageMIMEType(strings.ToLower(strings.TrimSpace(mimeType.String()))) {
			continue
		}
		data := imagePart.Get("data")
		isInline := true
		if key == "fileData" || key == "file_data" {
			isInline = false
			data = imagePart.Get("fileUri")
			if !data.Exists() {
				data = imagePart.Get("file_uri")
			}
		}
		// 只认带有实际数据或 URI 的图片 part；同一个 part 最多计费一次。
		if data.Type == gjson.String && strings.TrimSpace(data.String()) != "" {
			if isInline {
				return true, decodeGeminiImageDimensions(data.String())
			}
			return true, ""
		}
	}
	return false, ""
}

// decodeGeminiImageDimensions extracts "WIDTHxHEIGHT" from a base64 image header
// without decoding the full pixel payload whenever possible.
func decodeGeminiImageDimensions(rawBase64 string) string {
	b64 := strings.TrimSpace(rawBase64)
	if b64 == "" {
		return ""
	}
	if strings.HasPrefix(strings.ToLower(b64), "data:") {
		if idx := strings.Index(b64, ";base64,"); idx != -1 {
			b64 = strings.TrimSpace(b64[idx+len(";base64,"):])
		}
	}
	if len(b64) < 16 {
		return ""
	}
	const prefixB64Limit = 131072 // 98,304 bytes covers PNG/WebP/GIF and typical JPEG headers
	decodePart := b64
	truncated := false
	if len(decodePart) > prefixB64Limit {
		decodePart = decodePart[:prefixB64Limit-(prefixB64Limit%4)]
		truncated = true
	}
	decoded, err := base64.StdEncoding.DecodeString(decodePart)
	if err != nil {
		decoded, err = base64.RawStdEncoding.DecodeString(decodePart)
		if err != nil {
			return ""
		}
	}
	cfg, _, err := image.DecodeConfig(bytes.NewReader(decoded))
	if (errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF)) && truncated {
		if fullDecoded, fullErr := base64.StdEncoding.DecodeString(b64); fullErr == nil {
			cfg, _, err = image.DecodeConfig(bytes.NewReader(fullDecoded))
		}
	}
	if err != nil || cfg.Width <= 0 || cfg.Height <= 0 {
		return ""
	}
	return fmt.Sprintf("%dx%d", cfg.Width, cfg.Height)
}
