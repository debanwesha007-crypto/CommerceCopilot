"""
Deterministic fallback agent. No LLM call, no external dependency, no key
required — this is what runs by default and what runs if the LLM agent
errors out mid-demo. Intent matching is intentionally simple (keywords +
regex) rather than clever, because the point of this path is that it
*always* works.
"""
import re
from typing import Dict, Any, Optional
from app.agent import tools
from app.services.catalog_service import catalog_service
from app.services.cart_service import cart_service

CATEGORY_WORDS = {"laptop", "audio", "monitor", "furniture", "kitchen", "accessories", "home"}


def _extract_max_price(message: str) -> Optional[float]:
    m = re.search(r"under\s+(?:rs\.?|inr|₹)?\s*([\d,]+)", message, re.I)
    if not m:
        m = re.search(r"(?:below|less than)\s+(?:rs\.?|inr|₹)?\s*([\d,]+)", message, re.I)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def _extract_category(message: str) -> Optional[str]:
    lower = message.lower()
    for cat in CATEGORY_WORDS:
        if cat in lower:
            return cat
    return None


def _find_mentioned_product(message: str) -> Optional[Dict[str, Any]]:
    lower = message.lower()
    best = None
    for p in catalog_service.products:
        name_words = p["name"].lower().split()
        hits = sum(1 for w in name_words if w in lower and len(w) > 2)
        if hits >= 2 and (best is None or hits > best[1]):
            best = (p, hits)
    return best[0] if best else None


def handle_message(session_id: str, message: str) -> Dict[str, Any]:
    lower = message.lower().strip()

    # --- checkout confirmation flow takes priority ---
    if cart_service.is_pending_confirmation(session_id):
        if lower in ("yes", "confirm", "y", "confirm checkout", "proceed"):
            result = tools.checkout(session_id, confirmed=True)
            return _format_checkout_result(result)
        if lower in ("no", "cancel", "n"):
            cart_service.set_pending_confirmation(session_id, False)
            return {"reply": "Checkout cancelled — your cart is still saved.", "cart": cart_service.view(session_id)}
        # any other message while pending: fall through, but re-remind
        # (don't force it — user might be asking something else first)

    # --- checkout intent ---
    if any(k in lower for k in ["checkout", "buy now", "place order", "complete purchase"]):
        result = tools.checkout(session_id, confirmed=False)
        return _format_checkout_result(result)

    # --- add to cart ---
    if "cart" in lower and any(k in lower for k in ["add", "put"]):
        product = _find_mentioned_product(message)
        if not product:
            return {"reply": "Which product would you like to add? Try naming it, e.g. \"add AeroBook 14 to cart\"."}
        # Only treat a number as quantity if it's explicitly marked as one
        # (e.g. "qty 2", "x3", "2 units") — a bare number is too likely to be
        # part of the product name itself (e.g. "AeroBook 14").
        qty_match = re.search(r"(?:qty|quantity|x)\s*[:=]?\s*(\d+)\b", lower) or \
                    re.search(r"\b(\d+)\s*(?:units?|pieces?|pcs)\b", lower)
        qty = int(qty_match.group(1)) if qty_match else 1
        result = tools.add_to_cart(session_id, product["id"], qty)
        return {"reply": f"Added {qty} x {product['name']} to your cart. Current total: ₹{result['total']:.0f}.",
                "cart": result}

    # --- review Q&A ---
    if any(k in lower for k in ["review", "reviews say", "people say", "worth it", "battery life", "is it good"]):
        product = _find_mentioned_product(message)
        if not product:
            return {"reply": "Which product's reviews should I check? Name the product and your question."}
        result = tools.ask_reviews(session_id, product["id"], message)
        prefix = "" if result["grounded"] else "(Not grounded in reviews) "
        return {"reply": prefix + result["answer"]}

    # --- compare ---
    if "compare" in lower or " vs " in lower or " versus " in lower:
        # naive: compare all products whose names appear in the message
        mentioned = [p for p in catalog_service.products if p["name"].lower() in lower]
        if len(mentioned) < 2:
            # fall back: compare top matches from a search on the message
            candidates = catalog_service.search(message)
            ids = [c["id"] for c in candidates[:3]]
        else:
            ids = [p["id"] for p in mentioned]
        result = tools.compare_products(session_id, ids)
        if not result["products"]:
            return {"reply": "I couldn't find matching products to compare. Try naming two products directly."}
        lines = [f"{p['name']} — ₹{p['price']:.0f}, rating {p['rating']}" for p in result["products"]]
        return {"reply": "Here's the comparison:\n" + "\n".join(lines), "products": result["products"]}

    # --- default: search ---
    max_price = _extract_max_price(message)
    category = _extract_category(message)
    result = tools.search_products(session_id, message, max_price, category)
    products = result["products"]
    if not products:
        return {"reply": "I couldn't find anything matching that. Try a different keyword or a higher budget."}
    lines = [f"{p['name']} — ₹{p['price']:.0f} (rating {p['rating']})" for p in products[:5]]
    return {"reply": "Here's what I found:\n" + "\n".join(lines), "products": products}


def _format_checkout_result(result: Dict[str, Any]) -> Dict[str, Any]:
    status = result.get("status")
    if status == "confirmation_required":
        cart = result["cart"]
        lines = [f"{i['name']} x{i['qty']} — ₹{i['line_total']:.0f}" for i in cart["items"]]
        return {
            "reply": "Here's your order:\n" + "\n".join(lines) +
                     f"\nTotal: ₹{cart['total']:.0f}\n\nConfirm checkout via Razorpay test-mode? (yes/no)",
            "cart": cart, "pending_confirmation": True,
        }
    if status == "success":
        return {"reply": f"Payment captured. Order {result['order_id']} confirmed (test mode). "
                          f"Amount charged: ₹{result['amount']:.0f}."}
    if status == "failed":
        return {"reply": f"Payment failed ({result.get('reason', 'unknown error')}). "
                          f"No charge was made — your cart is still saved, want to retry?",
                "cart": result.get("cart")}
    if status == "rejected" and result.get("reason") == "cart_empty":
        return {"reply": "Your cart is empty — add something first."}
    if status == "rejected" and result.get("reason") == "exceeds_spend_cap":
        return {"reply": f"This order (₹{result['total']:.0f}) exceeds the ₹{result['cap']:.0f} session spend cap, "
                          f"so I can't check out automatically. Please remove an item or check out manually."}
    return {"reply": "Something unexpected happened during checkout."}
