"""
Optional LLM-driven agent. Only imported/constructed when USE_LLM_AGENT=true
and ANTHROPIC_API_KEY is set — the app must not fail to start without them.

This is where "AI judgment" actually earns its keep over the rule-based
path: freer-form questions, multi-step reasoning about which tool to call,
and more natural replies. The rule-based agent remains the fallback if this
raises for any reason (see orchestrator.py).
"""
from typing import Dict, Any, List
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from app.agent import tools as tool_impl
from app.config import ANTHROPIC_API_KEY, LLM_MODEL

SYSTEM_PROMPT = """You are CommerceCopilot, a shopping assistant for a merchant catalog.
Rules you must follow:
- Only state facts returned by tool calls. Never invent prices, specs, or review content.
- For any question about reviews or "what do people say", call ask_reviews rather than guessing.
- Never call checkout with confirmed=true unless the user has explicitly confirmed an order
  you already showed them (e.g. they say "yes", "confirm", "proceed"). Always show the cart
  and total first via an unconfirmed checkout call.
- Keep replies short and concrete: product names, prices, and citations, not filler."""


def _make_tools(session_id: str):
    @tool
    def search_products(query: str, max_price: float = None, category: str = None) -> Dict[str, Any]:
        """Search the merchant catalog by keyword, optional max price, optional category."""
        return tool_impl.search_products(session_id, query, max_price, category)

    @tool
    def get_product_details(product_id: str) -> Dict[str, Any]:
        """Get full specs and review snippets for one product by id."""
        return tool_impl.get_product_details(session_id, product_id)

    @tool
    def compare_products(product_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple products side by side by their ids."""
        return tool_impl.compare_products(session_id, product_ids)

    @tool
    def ask_reviews(product_id: str, question: str) -> Dict[str, Any]:
        """Ask a review-grounded question about one product. Returns grounded=false if unsupported."""
        return tool_impl.ask_reviews(session_id, product_id, question)

    @tool
    def add_to_cart(product_id: str, qty: int = 1) -> Dict[str, Any]:
        """Add a product to the session's cart."""
        return tool_impl.add_to_cart(session_id, product_id, qty)

    @tool
    def checkout(confirmed: bool = False) -> Dict[str, Any]:
        """Checkout the current cart via Razorpay test mode. Call with confirmed=false first
        to show the order summary; only call with confirmed=true after the user explicitly agrees."""
        return tool_impl.checkout(session_id, confirmed)

    return [search_products, get_product_details, compare_products, ask_reviews, add_to_cart, checkout]


def run_llm_agent(session_id: str, message: str, max_steps: int = 4) -> Dict[str, Any]:
    llm = ChatAnthropic(model=LLM_MODEL, api_key=ANTHROPIC_API_KEY, temperature=0)
    session_tools = _make_tools(session_id)
    llm_with_tools = llm.bind_tools(session_tools)
    tools_by_name = {t.name: t for t in session_tools}

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=message)]
    last_tool_result = None

    for _ in range(max_steps):
        ai_msg: AIMessage = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        if not ai_msg.tool_calls:
            return {"reply": ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content),
                    "last_tool_result": last_tool_result}
        for call in ai_msg.tool_calls:
            fn = tools_by_name[call["name"]]
            result = fn.invoke(call["args"])
            last_tool_result = {"tool": call["name"], "result": result}
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return {"reply": "I wasn't able to finish that within my step limit — could you rephrase or simplify the request?",
            "last_tool_result": last_tool_result}
