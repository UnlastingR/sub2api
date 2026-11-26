package service

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/ctxkey"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestOpenAIFastPolicyPlanRulesHTTPAndWS(t *testing.T) {
	settings := &OpenAIFastPolicySettings{Rules: []OpenAIFastPolicyRule{
		{ServiceTier: OpenAIFastTierAnyIncludingMissing, Action: OpenAIFastPolicyActionForcePriority, Scope: BetaPolicyScopeAll, UserIDs: []int64{42}},
		{ServiceTier: OpenAIFastTierAnyIncludingMissing, Action: OpenAIFastPolicyActionForceFlex, Scope: BetaPolicyScopeOAuth, PlanTypes: []string{"plus"}},
		{ServiceTier: OpenAIFastTierAnyIncludingMissing, Action: OpenAIFastPolicyActionForcePriority, Scope: BetaPolicyScopeOAuth, PlanTypes: []string{"team", "self_serve_business_usage_based", "self_serve_business_prolite"}},
	}}
	svc := newOpenAIGatewayServiceWithSettings(t, settings)
	for _, tc := range []struct {
		plan, want string
	}{
		{"Plus", "flex"},
		{"team", "priority"},
		{"self_serve_business_usage_based", "priority"},
		{"self-serve-business-prolite", "priority"},
		{"business", ""}, // distinct Enterprise SKU
		{"pro", ""},
		{"", ""},
	} {
		t.Run(tc.plan, func(t *testing.T) {
			account := &Account{Platform: PlatformOpenAI, Type: AccountTypeOAuth, Credentials: map[string]any{"plan_type": tc.plan}}
			for _, uid := range []int64{42, 43} {
				ctx := context.WithValue(context.Background(), ctxkey.UserID, uid)
				for _, input := range []string{"", "fast", "priority", "flex", "auto", "default", "scale", "ultrafast"} {
					want := tc.want
					if want == "" {
						if uid == 42 {
							want = "priority"
						} else if input == "fast" {
							want = "priority"
						} else {
							want = input
						}
					}
					payload := map[string]any{"type": "response.create", "model": "gpt-6.2-sol"}
					if input != "" {
						payload["service_tier"] = input
					}
					body, err := json.Marshal(payload)
					require.NoError(t, err)
					httpBody, err := svc.applyOpenAIFastPolicyToBody(ctx, account, "gpt-6.2-sol", body)
					require.NoError(t, err)
					require.Equal(t, want, gjson.GetBytes(httpBody, "service_tier").String(), "HTTP input=%q user=%d", input, uid)
					wsBody, blocked, err := svc.applyOpenAIFastPolicyToWSResponseCreate(ctx, account, "gpt-6.2-sol", body)
					require.NoError(t, err)
					require.Nil(t, blocked)
					require.Equal(t, want, gjson.GetBytes(wsBody, "service_tier").String(), "WS input=%q user=%d", input, uid)
				}
			}
		})
	}
}

func TestOpenAIFastPolicyPlanRuleOverridesGroupForcedFast(t *testing.T) {
	settings := &OpenAIFastPolicySettings{Rules: []OpenAIFastPolicyRule{
		{ServiceTier: OpenAIFastTierAnyIncludingMissing, Action: OpenAIFastPolicyActionForceFlex, Scope: BetaPolicyScopeAll, PlanTypes: []string{"plus"}},
	}}
	svc := newOpenAIGatewayServiceWithSettings(t, settings)
	ctx := context.WithValue(context.Background(), ctxkey.Group, &Group{ID: 7, Platform: PlatformOpenAI, Status: StatusActive, Hydrated: true, ForceOpenAIFast: true})
	account := &Account{Platform: PlatformOpenAI, Type: AccountTypeOAuth, Credentials: map[string]any{"plan_type": "plus"}}
	body := []byte(`{"type":"response.create","model":"gpt-6.2-sol"}`)
	httpBody, err := svc.applyOpenAIFastPolicyToBody(ctx, account, "gpt-6.2-sol", body)
	require.NoError(t, err)
	require.Equal(t, "flex", gjson.GetBytes(httpBody, "service_tier").String())
	wsBody, blocked, err := svc.applyOpenAIFastPolicyToWSResponseCreate(ctx, account, "gpt-6.2-sol", body)
	require.NoError(t, err)
	require.Nil(t, blocked)
	require.Equal(t, "flex", gjson.GetBytes(wsBody, "service_tier").String())
	// API-key accounts must never match subscription rules, even with a plan_type field.
	account.Type = AccountTypeAPIKey
	httpBody, err = svc.applyOpenAIFastPolicyToBody(ctx, account, "gpt-6.2-sol", body)
	require.NoError(t, err)
	require.Equal(t, "priority", gjson.GetBytes(httpBody, "service_tier").String())
}

func TestSetOpenAIFastPolicyPlanSettingsRoundTrip(t *testing.T) {
	svc := NewSettingService(&openAIFastPolicyRepoStub{values: map[string]string{}}, &config.Config{})
	ctx := context.Background()
	settings := &OpenAIFastPolicySettings{Rules: []OpenAIFastPolicyRule{
		{ServiceTier: OpenAIFastTierAnyIncludingMissing, Action: OpenAIFastPolicyActionForceFlex, Scope: BetaPolicyScopeOAuth, PlanTypes: []string{" Plus ", "team"}},
	}}
	require.NoError(t, svc.SetOpenAIFastPolicySettings(ctx, settings))
	got, err := svc.GetOpenAIFastPolicySettings(ctx)
	require.NoError(t, err)
	require.Equal(t, []string{"plus", "team"}, got.Rules[0].PlanTypes)
	require.Equal(t, OpenAIFastPolicyActionForceFlex, got.Rules[0].Action)
	require.Equal(t, OpenAIFastTierAnyIncludingMissing, got.Rules[0].ServiceTier)
	settings.Rules[0].PlanTypes = []string{"team", "TEAM"}
	require.Error(t, svc.SetOpenAIFastPolicySettings(ctx, settings))
	settings.Rules[0].PlanTypes = []string{" "}
	require.Error(t, svc.SetOpenAIFastPolicySettings(ctx, settings))
}
