package service

import "strings"

// openAIWSIngressEffortState is private to one downstream ingress session.
// Responses clients can retain a top-level baseline for caching while a
// configuration_update in the response chain controls the effective effort.
// Do not persist this state in the pool or use it to rewrite an outbound frame.
type openAIWSIngressEffortState struct {
	connID, responseID, model string
	effective, requested      string
}

func (s *openAIWSIngressEffortState) resolve(
	connID, previousID, responseID, mappedModel, originalModel string,
	outbound, inbound []byte, requested *string, completed bool,
) (effective, requestedEffective *string) {
	effective = ApplyThinkingEnabledFallback(extractOpenAIReasoningEffortFromBody(outbound, mappedModel, originalModel), outbound, mappedModel)
	requestedEffective = requested
	model := strings.ToLower(strings.TrimSpace(firstNonEmpty(mappedModel, originalModel)))
	// An explicit reference to a different history must not inherit our state.
	// No reference is also possible on a subsequent turn on the same lease.
	next := openAIWSIngressEffortState{}
	if connID != "" && model != "" && s.connID == connID && s.model == model &&
		(previousID == "" || previousID == s.responseID) {
		next = *s
	}

	rawUpdate := latestResponsesInputEffort(inbound)
	if update := latestResponsesInputEffort(outbound); update != "" {
		next.effective = normalizeOpenAIReasoningEffortForModel(update, firstNonEmpty(mappedModel, originalModel))
		if rawUpdate != "" {
			next.requested = normalizeReasoningEffortMappingSource(rawUpdate)
		} else {
			// This can be an update restored from the prior outbound history.
			next.requested = firstNonEmpty(next.requested, normalizeReasoningEffortMappingSource(update))
		}
	}
	if next.effective != "" {
		effective = optionalTrimmedStringPtr(next.effective)
	}
	if rawUpdate == "" && next.requested != "" {
		requestedEffective = optionalTrimmedStringPtr(next.requested)
	}

	// Failed attempts, including retries, do not teach the session new state.
	if completed && connID != "" && model != "" {
		next.connID, next.responseID, next.model = connID, responseID, model
		*s = next
	}
	return effective, requestedEffective
}
