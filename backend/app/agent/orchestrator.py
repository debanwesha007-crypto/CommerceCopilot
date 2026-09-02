from typing import Dict, Any
from app.agent import rule_based_agent
from app.services.cart_service import cart_service
from app.config import USE_LLM_AGENT, ANTHROPIC_API_KEY


def handle_message(session_id: str, message: str) -> Dict[str, Any]:
    if USE_LLM_AGENT and ANTHROPIC_API_KEY:
        try:
            from app.agent.llm_agent import run_llm_agent  # imported lazily: keeps the
            # rule-based path importable even if langchain_anthropic isn't installed
            result = run_llm_agent(session_id, message)
            return {
                "reply": result["reply"],
                "cart": cart_service.view(session_id),
                "pending_confirmation": cart_service.is_pending_confirmation(session_id),
            }
        except Exception as e:
            # Documented failure-recovery path: an LLM/API error never breaks the
            # conversation, it silently drops to the deterministic agent instead.
            fallback = rule_based_agent.handle_message(session_id, message)
            fallback["reply"] = f"[LLM agent unavailable ({type(e).__name__}), used fallback] " + fallback["reply"]
            return fallback

    return rule_based_agent.handle_message(session_id, message)
