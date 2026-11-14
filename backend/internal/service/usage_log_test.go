package service

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestParseUsageRequestType(t *testing.T) {
	t.Parallel()

	type testCase struct {
		name    string
		input   string
		want    RequestType
		wantErr bool
	}

	cases := []testCase{
		{name: "unknown", input: "unknown", want: RequestTypeUnknown},
		{name: "sync", input: "sync", want: RequestTypeSync},
		{name: "stream", input: "stream", want: RequestTypeStream},
		{name: "ws_v2", input: "ws_v2", want: RequestTypeWSV2},
		{name: "case_insensitive", input: "WS_V2", want: RequestTypeWSV2},
		{name: "trim_spaces", input: "  stream  ", want: RequestTypeStream},
		{name: "batch", input: "batch", want: RequestTypeBatch},
		{name: "invalid", input: "xxx", wantErr: true},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := ParseUsageRequestType(tc.input)
			if tc.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.want, got)
		})
	}
}

func TestRequestTypeNormalizeAndString(t *testing.T) {
	t.Parallel()

	require.Equal(t, RequestTypeUnknown, RequestType(99).Normalize())
	require.Equal(t, "unknown", RequestType(99).String())
	require.Equal(t, "sync", RequestTypeSync.String())
	require.Equal(t, "stream", RequestTypeStream.String())
	require.Equal(t, "ws_v2", RequestTypeWSV2.String())
	require.Equal(t, "batch", RequestTypeBatch.String())
	require.True(t, RequestTypeBatch.IsValid())
}

func TestGeminiBatchRequestTypeForExistingUsage(t *testing.T) {
	t.Parallel()

	for _, endpoint := range []string{
		"/v1beta/models/gemini-2.5-flash-lite:batchGenerateContent",
		"/v1beta/models/gemini-embedding-001:asyncBatchEmbedContent",
		"/v1beta/models/gemini-embedding-001:asyncBatchEmbedContents",
	} {
		require.True(t, isGeminiBatchUsageEndpoint(endpoint))
		for _, previousType := range []RequestType{RequestTypeSync, RequestTypeUnknown, RequestTypeBatch} {
			log := &UsageLog{RequestType: previousType, InboundEndpoint: &endpoint}
			require.Equal(t, RequestTypeBatch, log.EffectiveRequestType())
		}
	}
	for _, endpoint := range []string{"/v1beta/models", "/v1beta/models/gemini-2.5-flash-lite:generateContent", "/v1beta/decisions", "/v1beta/batches"} {
		require.False(t, isGeminiBatchUsageEndpoint(endpoint))
		log := &UsageLog{RequestType: RequestTypeSync, InboundEndpoint: &endpoint}
		require.Equal(t, RequestTypeSync, log.EffectiveRequestType())
	}
	endpoint := "/v1beta/models/gemini-2.5-flash-lite:batchGenerateContent"
	cyber := &UsageLog{RequestType: RequestTypeCyberBlocked, InboundEndpoint: &endpoint}
	require.Equal(t, RequestTypeCyberBlocked, cyber.EffectiveRequestType())

	batch := &UsageLog{RequestType: RequestTypeBatch, Stream: true, OpenAIWSMode: true}
	batch.SyncRequestTypeAndLegacyFields()
	require.False(t, batch.Stream)
	require.False(t, batch.OpenAIWSMode)
}

func TestRequestTypeFromLegacy(t *testing.T) {
	t.Parallel()

	require.Equal(t, RequestTypeWSV2, RequestTypeFromLegacy(false, true))
	require.Equal(t, RequestTypeStream, RequestTypeFromLegacy(true, false))
	require.Equal(t, RequestTypeSync, RequestTypeFromLegacy(false, false))
}

func TestApplyLegacyRequestFields(t *testing.T) {
	t.Parallel()

	stream, ws := ApplyLegacyRequestFields(RequestTypeSync, true, true)
	require.False(t, stream)
	require.False(t, ws)

	stream, ws = ApplyLegacyRequestFields(RequestTypeStream, false, true)
	require.True(t, stream)
	require.False(t, ws)

	stream, ws = ApplyLegacyRequestFields(RequestTypeWSV2, false, false)
	require.True(t, stream)
	require.True(t, ws)

	stream, ws = ApplyLegacyRequestFields(RequestTypeUnknown, true, false)
	require.True(t, stream)
	require.False(t, ws)
}

func TestUsageLogSyncRequestTypeAndLegacyFields(t *testing.T) {
	t.Parallel()

	log := &UsageLog{RequestType: RequestTypeWSV2, Stream: false, OpenAIWSMode: false}
	log.SyncRequestTypeAndLegacyFields()

	require.Equal(t, RequestTypeWSV2, log.RequestType)
	require.True(t, log.Stream)
	require.True(t, log.OpenAIWSMode)
}

func TestUsageLogEffectiveRequestTypeFallback(t *testing.T) {
	t.Parallel()

	log := &UsageLog{RequestType: RequestTypeUnknown, Stream: true, OpenAIWSMode: true}
	require.Equal(t, RequestTypeWSV2, log.EffectiveRequestType())
}

func TestUsageLogEffectiveRequestTypeNilReceiver(t *testing.T) {
	t.Parallel()

	var log *UsageLog
	require.Equal(t, RequestTypeUnknown, log.EffectiveRequestType())
}

func TestUsageLogSyncRequestTypeAndLegacyFieldsNilReceiver(t *testing.T) {
	t.Parallel()

	var log *UsageLog
	log.SyncRequestTypeAndLegacyFields()
}
