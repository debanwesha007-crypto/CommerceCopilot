"""
Agent-facing tools, matching the MCP tool schema from the architecture deck:
search_products, get_product_details, compare_products, ask_reviews,
add_to_cart, checkout.

Kept as plain Python functions with a JSON-serializable in/out contract so
they can be wrapped as LangChain tools, exposed over an MCP server (e.g.
NitroStack), or called directly from the rule-based agent — same functions,
three different callers.

Every call is audit-logged. checkout() additionally enforces a spend cap
and refuses to run without prior confirmation — the guardrails described
on the "AI judgment" slide, enforced in code rather than left to prompting.
"""
from typing import List, Dict, Any
from app.services.catalog_service import catalog_service
from app.services.rag_service import rag_service
from app.services.cart_service import cart_service
from app.services.razorpay_service import razorpay_client
from app.services.audit_service import audit_service
from app.config import MAX_CHECKOUT_AMOUNT


def search_products(session_id: str, query: str, max_price: float = None, category: str = None) -> Dict[str, Any]:
    result = {"products": catalog_service.search(query, max_price, category)}
    audit_service.log(session_id, "search_products", {"query": query, "max_price": max_price, "category": category}, result)
    return result


def get_product_details(session_id: str, product_id: str) -> Dict[str, Any]:
    details = catalog_service.get_details(product_id)
    result = details or {"error": "product_not_found"}
    audit_service.log(session_id, "get_product_details", {"product_id": product_id}, result)
    return result


def compare_products(session_id: str, product_ids: List[str]) -> Dict[str, Any]:
    result = catalog_service.compare(product_ids)
    audit_service.log(session_id, "compare_products", {"product_ids": product_ids}, result)
    return result


def ask_reviews(session_id: str, product_id: str, question: str) -> Dict[str, Any]:
    result = rag_service.ask(product_id, question)
    audit_service.log(session_id, "ask_reviews", {"product_id": product_id, "question": question}, result)
    return result


def add_to_cart(session_id: str, product_id: str, qty: int = 1) -> Dict[str, Any]:
    product = catalog_service.get_raw(product_id)
    if not product:
        result = {"error": "product_not_found"}
    else:
        result = cart_service.add(session_id, product_id, qty)
    audit_service.log(session_id, "add_to_cart", {"product_id": product_id, "qty": qty}, result)
    return result


def checkout(session_id: str, confirmed: bool = False) -> Dict[str, Any]:
    """Bounded + gated: refuses to run without explicit confirmation, and
    refuses above the configured spend cap. This is the tool call the
    architecture doc's audit trail exists to explain."""
    cart = cart_service.view(session_id)

    if not cart["items"]:
        result = {"status": "rejected", "reason": "cart_empty"}
        audit_service.log(session_id, "checkout", {"confirmed": confirmed}, result)
        return result

    if not confirmed:
        cart_service.set_pending_confirmation(session_id, True)
        result = {"status": "confirmation_required", "cart": cart}
        audit_service.log(session_id, "checkout", {"confirmed": confirmed}, result)
        return result

    if cart["total"] > MAX_CHECKOUT_AMOUNT:
        result = {"status": "rejected", "reason": "exceeds_spend_cap", "cap": MAX_CHECKOUT_AMOUNT, "total": cart["total"]}
        audit_service.log(session_id, "checkout", {"confirmed": confirmed}, result)
        return result

    order = razorpay_client.create_order(cart["total"])
    payment = razorpay_client.capture_payment(order["id"], cart["total"])

    if payment["status"] != "captured":
        # Explicit rollback: order was created but never confirmed as paid,
        # so nothing is charged. Cart is preserved for the user to retry.
        result = {"status": "failed", "reason": payment.get("error", "payment_failed"),
                   "order_id": order["id"], "cart": cart}
        audit_service.log(session_id, "checkout", {"confirmed": confirmed}, result)
        return result

    cart_service.clear(session_id)
    result = {"status": "success", "order_id": order["id"], "payment_id": payment["payment_id"],
              "amount": payment["amount"]}
    audit_service.log(session_id, "checkout", {"confirmed": confirmed}, result)
    return result


TOOL_REGISTRY = {
    "search_products": search_products,
    "get_product_details": get_product_details,
    "compare_products": compare_products,
    "ask_reviews": ask_reviews,
    "add_to_cart": add_to_cart,
    "checkout": checkout,
}
