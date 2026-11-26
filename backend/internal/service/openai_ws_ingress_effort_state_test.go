package service

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestOpenAIWSIngressEffortState(t *testing.T) {
	baseline := []byte(`{"reasoning":{"effort":"medium"}}`)
	updated := func(effort string) []byte {
		return []byte(fmt.Sprintf(`{"reasoning":{"effort":"medium"},"input":[{"type":"configuration_update","reasoning":{"effort":%q}}]}`, effort))
	}
	const sol = "gpt-6.2-sol"
	const astra = "gpt-6.2-astra"
	for _, tc := range []struct {
		name, conn, prev, model, want string
	}{
		{"chained", "conn-1", "resp_1", sol, "xhigh"},
		{"same session without explicit previous ID", "conn-1", "", sol, "xhigh"},
		{"different history", "conn-1", "resp_other", sol, "medium"},
		{"new upstream lease", "conn-2", "resp_1", sol, "medium"},
		{"different model", "conn-1", "resp_1", astra, "medium"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var state openAIWSIngressEffortState
			first := updated("xhigh")
			state.resolve("conn-1", "", "resp_1", sol, sol, first, first, CanonicalRequestedReasoningEffort(first), true)
			got, requested := state.resolve(tc.conn, tc.prev, "resp_2", tc.model, tc.model, baseline, baseline, CanonicalRequestedReasoningEffort(baseline), true)
			require.Equal(t, tc.want, optionalStringValue(got))
			require.Equal(t, tc.want, optionalStringValue(requested))
		})
	}

	var state openAIWSIngressEffortState
	run := func(prev, id string, payload []byte, success bool) string {
		got, _ := state.resolve("conn-1", prev, id, sol, sol, payload, payload, CanonicalRequestedReasoningEffort(payload), success)
		return optionalStringValue(got)
	}
	require.Equal(t, "xhigh", run("", "resp_1", updated("xhigh"), true))
	require.Equal(t, "low", run("resp_1", "resp_failed", updated("low"), false))
	require.Equal(t, "xhigh", run("resp_1", "resp_2", baseline, true))
	require.Equal(t, "low", run("resp_2", "resp_3", updated("low"), true))
	require.Equal(t, "low", run("resp_3", "resp_4", baseline, true))
	require.Equal(t, "medium", run("resp_4", "resp_5", updated("medium"), true))
	require.Equal(t, "medium", run("resp_5", "resp_6", baseline, true))
}
