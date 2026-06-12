//! Response decoding: Converse body → canonical `Response`.

use serde_json::Value;

use crate::codec::CodecCtx;
use crate::error::Error;
use crate::types::{
    ContentPart, FinishReason, Message, RateLimitInfo, Response, Role, ThinkingData, TokenCounts,
    ToolCall,
};

pub(super) fn decode_response(
    body: &str,
    ctx: &CodecCtx<'_>,
    rate_limit: Option<RateLimitInfo>,
) -> Result<Response, Error> {
    let raw: Value = serde_json::from_str(body)
        .map_err(|e| Error::network(format!("failed to parse converse response: {e}"), e))?;

    let content_parts = raw
        .pointer("/output/message/content")
        .and_then(Value::as_array)
        .map(|blocks| blocks.iter().filter_map(decode_content_block).collect())
        .unwrap_or_default();

    let finish_reason = map_stop_reason(raw.get("stopReason").and_then(Value::as_str));
    let usage = token_counts_from_usage(raw.get("usage"));

    Ok(Response {
        // Converse responses carry no id; synthesize one like the gemini
        // codec does so downstream consumers always see a non-empty id.
        id: uuid::Uuid::new_v4().to_string(),
        model: ctx.request.model.clone(),
        provider: ctx.provider_name.to_string(),
        message: Message {
            role:         Role::Assistant,
            content:      content_parts,
            name:         None,
            tool_call_id: None,
        },
        finish_reason,
        usage,
        raw: Some(raw),
        warnings: vec![],
        rate_limit,
        cost_usd: None,
        cost_source: None,
    })
}

/// Decode one Converse content block into a canonical part. Unknown block
/// kinds are skipped (the union grows: `citationsContent`, `searchResult`,
/// `video`, ...).
pub(super) fn decode_content_block(block: &Value) -> Option<ContentPart> {
    if let Some(text) = block.get("text").and_then(Value::as_str) {
        if text.is_empty() {
            return None;
        }
        return Some(ContentPart::text(text));
    }
    if let Some(tool_use) = block.get("toolUse") {
        let id = tool_use.get("toolUseId").and_then(Value::as_str)?;
        let name = tool_use.get("name").and_then(Value::as_str)?;
        let input = tool_use.get("input").cloned().unwrap_or(Value::Null);
        return Some(ContentPart::ToolCall(ToolCall::new(id, name, input)));
    }
    if let Some(reasoning) = block.get("reasoningContent") {
        if let Some(text_block) = reasoning.get("reasoningText") {
            return Some(ContentPart::Thinking(ThinkingData {
                text:      text_block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                signature: text_block
                    .get("signature")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                redacted:  false,
            }));
        }
        if let Some(redacted) = reasoning.get("redactedContent").and_then(Value::as_str) {
            return Some(ContentPart::Thinking(ThinkingData {
                text:      redacted.to_string(),
                signature: None,
                redacted:  true,
            }));
        }
    }
    None
}

/// Map a Converse `stopReason` onto the canonical finish vocabulary.
pub(super) fn map_stop_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        None | Some("end_turn" | "stop_sequence") => FinishReason::Stop,
        Some("max_tokens" | "model_context_window_exceeded") => FinishReason::Length,
        Some("tool_use") => FinishReason::ToolCalls,
        // `refusal` is the Claude 5 blocking-classifier stop, passed through
        // by Bedrock for Fable-class models.
        Some("guardrail_intervened" | "content_filtered" | "refusal") => {
            FinishReason::ContentFilter
        }
        Some(other) => FinishReason::Other(other.to_string()),
    }
}

/// Converse usage maps directly onto the disjoint buckets: `inputTokens`
/// already excludes cached tokens (documented), so no subtraction applies.
pub(super) fn token_counts_from_usage(usage: Option<&Value>) -> TokenCounts {
    let Some(usage) = usage else {
        return TokenCounts::default();
    };
    let count = |key: &str| usage.get(key).and_then(Value::as_i64).unwrap_or(0);
    TokenCounts {
        input_tokens:       count("inputTokens"),
        output_tokens:      count("outputTokens"),
        reasoning_tokens:   0,
        cache_read_tokens:  count("cacheReadInputTokens"),
        cache_write_tokens: count("cacheWriteInputTokens"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_reasons_map_to_canonical_vocabulary() {
        assert_eq!(map_stop_reason(Some("end_turn")), FinishReason::Stop);
        assert_eq!(map_stop_reason(Some("stop_sequence")), FinishReason::Stop);
        assert_eq!(map_stop_reason(Some("max_tokens")), FinishReason::Length);
        assert_eq!(
            map_stop_reason(Some("model_context_window_exceeded")),
            FinishReason::Length
        );
        assert_eq!(map_stop_reason(Some("tool_use")), FinishReason::ToolCalls);
        assert_eq!(
            map_stop_reason(Some("guardrail_intervened")),
            FinishReason::ContentFilter
        );
        assert_eq!(
            map_stop_reason(Some("content_filtered")),
            FinishReason::ContentFilter
        );
        assert_eq!(
            map_stop_reason(Some("refusal")),
            FinishReason::ContentFilter
        );
        assert_eq!(
            map_stop_reason(Some("malformed_tool_use")),
            FinishReason::Other("malformed_tool_use".to_string())
        );
        assert_eq!(map_stop_reason(None), FinishReason::Stop);
    }

    #[test]
    fn usage_maps_without_subtraction() {
        let usage = serde_json::json!({
            "inputTokens": 30,
            "outputTokens": 628,
            "totalTokens": 658,
            "cacheReadInputTokens": 1024,
            "cacheWriteInputTokens": 512,
        });
        let counts = token_counts_from_usage(Some(&usage));
        assert_eq!(counts.input_tokens, 30);
        assert_eq!(counts.output_tokens, 628);
        assert_eq!(counts.cache_read_tokens, 1024);
        assert_eq!(counts.cache_write_tokens, 512);
        assert_eq!(counts.reasoning_tokens, 0);
    }

    #[test]
    fn unknown_content_blocks_are_skipped() {
        assert!(decode_content_block(&serde_json::json!({"citationsContent": {}})).is_none());
        assert!(decode_content_block(&serde_json::json!({"text": ""})).is_none());
    }

    #[test]
    fn reasoning_text_block_round_trips_signature() {
        let block = serde_json::json!({
            "reasoningContent": {
                "reasoningText": { "text": "thinking...", "signature": "sig-1" }
            }
        });
        let Some(ContentPart::Thinking(thinking)) = decode_content_block(&block) else {
            panic!("expected thinking part");
        };
        assert_eq!(thinking.text, "thinking...");
        assert_eq!(thinking.signature.as_deref(), Some("sig-1"));
        assert!(!thinking.redacted);
    }
}
