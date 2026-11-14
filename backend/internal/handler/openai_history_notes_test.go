package handler

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	middleware2 "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type historyNotesAccountRepoStub struct {
	service.AccountRepository
}

type historyNotesCacheStub struct {
	service.GatewayCache
	err error
}

var _ service.AccountRepository = (*historyNotesAccountRepoStub)(nil)
var _ service.GatewayCache = (*historyNotesCacheStub)(nil)

func (s *historyNotesCacheStub) GetSessionAccountID(context.Context, int64, string) (int64, error) {
	return 0, s.err
}

func newHistoryNotesTestHandler(t *testing.T, cacheErr error) *OpenAIGatewayHandler {
	t.Helper()

	gatewayService := service.NewOpenAIGatewayService(
		&historyNotesAccountRepoStub{},
		nil, nil, nil, nil, nil,
		&historyNotesCacheStub{err: cacheErr},
		nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil,
	)
	t.Cleanup(gatewayService.StopOpenAICodexTicketHarvester)

	return &OpenAIGatewayHandler{gatewayService: gatewayService}
}

func newHistoryNotesTestContext(namespace, action string) (*gin.Context, *httptest.ResponseRecorder) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	path := "/v1/alpha/" + namespace + "/v2/" + action
	c.Request = httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"context":{"session_id":"codex-session-123"}}`))
	c.Request.Header.Set("Content-Type", "application/json")
	c.Params = gin.Params{
		{Key: "state_namespace", Value: namespace},
		{Key: "action", Value: action},
	}
	groupID := int64(4)
	c.Set(string(middleware2.ContextKeyAPIKey), &service.APIKey{
		ID:      1,
		GroupID: &groupID,
		Group:   &service.Group{ID: groupID, Platform: service.PlatformOpenAI},
	})
	return c, recorder
}

func TestHistoryNotes_StickySessionMiss(t *testing.T) {
	handler := newHistoryNotesTestHandler(t, service.ErrStickySessionNotFound)

	tests := []struct {
		name      string
		namespace string
		action    string
		wantCode  int
		wantBody  string
	}{
		{
			name:      "new thread hint returns an empty hint",
			namespace: "notes",
			action:    "thread_hint",
			wantCode:  http.StatusOK,
			wantBody:  `{"text":""}`,
		},
		{
			name:      "other history calls remain unbound conflicts",
			namespace: "history",
			action:    "list_items",
			wantCode:  http.StatusConflict,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, recorder := newHistoryNotesTestContext(tt.namespace, tt.action)
			handler.HistoryNotes(c)

			require.Equal(t, tt.wantCode, recorder.Code, recorder.Body.String())
			if tt.wantBody != "" {
				require.JSONEq(t, tt.wantBody, recorder.Body.String())
			} else {
				require.Contains(t, recorder.Body.String(), "session_unavailable")
			}
		})
	}
}

func TestHistoryNotes_StickySessionCacheFailureRemainsBadGateway(t *testing.T) {
	cacheErr := errors.New("redis connection refused")
	handler := newHistoryNotesTestHandler(t, cacheErr)
	c, recorder := newHistoryNotesTestContext("notes", "thread_hint")

	handler.HistoryNotes(c)

	require.Equal(t, http.StatusBadGateway, recorder.Code, recorder.Body.String())
	require.Contains(t, recorder.Body.String(), "upstream_error")
}
