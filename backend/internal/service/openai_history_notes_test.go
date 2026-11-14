package service

import (
	"context"
	"errors"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/stretchr/testify/require"
)

type codexHistoryNotesCacheStub struct {
	*stubGatewayCache
	accountID int64
	lookupErr error
}

func (c *codexHistoryNotesCacheStub) GetSessionAccountID(_ context.Context, _ int64, _ string) (int64, error) {
	return c.accountID, c.lookupErr
}

func TestCodexHistoryNotesStickyAccountID(t *testing.T) {
	transportErr := errors.New("redis connection lost")
	tests := []struct {
		name      string
		cache     GatewayCache
		accountID int64
		wantErr   error
	}{
		{"cache missing", nil, 0, ErrCodexHistoryNotesSessionUnbound},
		{"redis key absent", &codexHistoryNotesCacheStub{lookupErr: ErrStickySessionNotFound}, 0, ErrCodexHistoryNotesSessionUnbound},
		{"empty binding", &codexHistoryNotesCacheStub{}, 0, ErrCodexHistoryNotesSessionUnbound},
		{"existing binding", &codexHistoryNotesCacheStub{accountID: 42}, 42, nil},
		{"redis failure", &codexHistoryNotesCacheStub{lookupErr: transportErr}, 0, transportErr},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			svc := &OpenAIGatewayService{
				cache: tc.cache,
				cfg: &config.Config{Gateway: config.GatewayConfig{
					OpenAIWS: config.GatewayOpenAIWSConfig{SessionHashReadOldFallback: false},
				}},
			}
			id, err := svc.codexHistoryNotesStickyAccountID(context.Background(), nil, "test-codex-session")
			require.Equal(t, tc.accountID, id)
			if tc.wantErr == nil {
				require.NoError(t, err)
			} else {
				require.ErrorIs(t, err, tc.wantErr)
			}
			if tc.wantErr == transportErr {
				require.NotErrorIs(t, err, ErrCodexHistoryNotesSessionUnbound)
			}
		})
	}
}
