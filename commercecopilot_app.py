"""
CommerceCopilot — single-file prototype
Razorpay AI Buildathon — Agentic Commerce track

Run:
    pip install fastapi uvicorn pydantic scikit-learn numpy
    python app.py
Then open http://localhost:8000 in a browser. No npm, no build step,
no API keys required — runs fully offline with a rule-based agent and
a mocked Razorpay test-mode client.

Optional LLM agent: pip install anthropic, then set
USE_LLM_AGENT=true and ANTHROPIC_API_KEY=sk-ant-... as env vars.

Optional live Razorpay test mode: pip install razorpay, then set
RAZORPAY_LIVE=true, RAZORPAY_KEY_ID=..., RAZORPAY_KEY_SECRET=...
"""

import json
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

# =============================================================================
# CONFIG
# =============================================================================

USE_LLM_AGENT = os.getenv("USE_LLM_AGENT", "false").lower() == "true"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

RAZORPAY_LIVE = os.getenv("RAZORPAY_LIVE", "false").lower() == "true"
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")

MAX_CHECKOUT_AMOUNT = int(os.getenv("MAX_CHECKOUT_AMOUNT", "100000"))
SIMULATED_PAYMENT_FAILURE_RATE = float(os.getenv("SIMULATED_PAYMENT_FAILURE_RATE", "0.1"))
GROUNDING_THRESHOLD = 0.08

AUDIT_LOG_PATH = Path(__file__).resolve().parent / "audit_log.jsonl"

# =============================================================================
# CATALOG DATA (seeded — swap for a real merchant catalog API)
# =============================================================================

CATALOG: List[Dict[str, Any]] = [
    {"id": "p001", "name": "AeroBook 14 Slim Laptop", "category": "laptop", "price": 42999, "rating": 4.3,
     "specs": {"cpu": "8-core, 3.2GHz", "ram": "16GB", "storage": "512GB SSD", "battery": "Up to 14 hours", "weight": "1.2kg"},
     "reviews": [
         "Battery genuinely lasts a full workday, closer to 12 hours with normal browsing and docs.",
         "Fan noise is noticeable under heavy load but silent for everyday tasks.",
         "Build feels premium for the price, aluminum body doesn't flex.",
         "Screen brightness is good indoors but washes out a bit in direct sunlight.",
         "Shipped with minor bloatware, easy to remove though."]},
    {"id": "p002", "name": "AeroBook 14 Slim Laptop - Pro", "category": "laptop", "price": 54999, "rating": 4.5,
     "specs": {"cpu": "8-core, 3.6GHz", "ram": "32GB", "storage": "1TB SSD", "battery": "Up to 13 hours", "weight": "1.25kg"},
     "reviews": [
         "Noticeably snappier than the base model when running multiple heavy apps at once.",
         "Battery is slightly shorter than the base model because of the faster chip, still lasts most of a workday.",
         "Worth the upgrade if you do any video editing or run local AI models.",
         "Same chassis as the base model, so no difference in build quality or weight."]},
    {"id": "p003", "name": "ValueBook 15 Everyday Laptop", "category": "laptop", "price": 28999, "rating": 3.9,
     "specs": {"cpu": "4-core, 2.4GHz", "ram": "8GB", "storage": "256GB SSD", "battery": "Up to 8 hours", "weight": "1.6kg"},
     "reviews": [
         "Good budget option for browsing, docs, and video calls, struggles with anything heavier.",
         "Battery drains fast if you keep more than a few browser tabs open.",
         "Plastic build feels a bit cheap but hasn't broken after 6 months of daily use.",
         "8GB RAM is the real bottleneck, upgrade to 16GB if your budget allows."]},
    {"id": "p004", "name": "TrailRunner Wireless Earbuds", "category": "audio", "price": 3499, "rating": 4.1,
     "specs": {"battery": "6h + 24h case", "waterproof": "IPX5", "connectivity": "Bluetooth 5.3"},
     "reviews": [
         "Stay in place well during running and gym sessions, never had them fall out.",
         "Bass is decent but mids feel a little recessed compared to pricier earbuds.",
         "Case charging is fast, went from empty to full in under an hour.",
         "Touch controls are a bit too sensitive, accidentally paused a few times."]},
    {"id": "p005", "name": "StudioTone ANC Headphones", "category": "audio", "price": 8999, "rating": 4.6,
     "specs": {"battery": "30h with ANC on", "type": "Over-ear", "connectivity": "Bluetooth 5.2, wired option"},
     "reviews": [
         "ANC is genuinely excellent, blocks out most office and flight noise.",
         "Comfortable enough for 4-5 hour sessions, ear cups don't get hot.",
         "Sound is well balanced, good for both music and calls.",
         "A bit bulky to carry around, the included case is large."]},
    {"id": "p006", "name": "PixelClear 27 Monitor", "category": "monitor", "price": 15999, "rating": 4.4,
     "specs": {"size": "27 inch", "resolution": "2560x1440", "refresh": "75Hz", "panel": "IPS"},
     "reviews": [
         "Colors are accurate out of the box, minimal calibration needed for design work.",
         "75Hz is a nice bump from 60Hz for everyday use, not marketed as a gaming monitor though.",
         "Stand adjustability is good, height and tilt cover most desk setups.",
         "Two units had minor backlight bleed in corners, worth checking on arrival."]},
    {"id": "p007", "name": "PixelClear 27 Monitor - Gaming Edition", "category": "monitor", "price": 21999, "rating": 4.5,
     "specs": {"size": "27 inch", "resolution": "2560x1440", "refresh": "165Hz", "panel": "IPS"},
     "reviews": [
         "165Hz makes a real difference in fast-paced games, motion is much smoother.",
         "Same panel quality as the standard PixelClear 27 for color accuracy.",
         "Response time is good, minimal ghosting even in competitive shooters.",
         "Runs a bit warmer than the standard model during long gaming sessions."]},
    {"id": "p008", "name": "DeskMate Ergonomic Chair", "category": "furniture", "price": 12499, "rating": 4.0,
     "specs": {"material": "Mesh back, foam seat", "adjustments": "Height, armrest, recline", "weight_capacity": "120kg"},
     "reviews": [
         "Lower back support made a noticeable difference after a week of daily use.",
         "Assembly took about 30 minutes, instructions were clear.",
         "Armrests feel a bit wobbly compared to the rest of the build.",
         "Mesh back breathes well, no more sweaty back during long sessions."]},
    {"id": "p009", "name": "QuickBrew Espresso Machine", "category": "kitchen", "price": 9499, "rating": 4.2,
     "specs": {"pressure": "15 bar", "capacity": "1.2L water tank", "milk_frother": "Built-in steam wand"},
     "reviews": [
         "Pulls a consistent shot once you dial in the grind size, takes some experimenting at first.",
         "Steam wand is manual, has a learning curve for latte art but works well once you get it.",
         "Compact enough for a small kitchen counter.",
         "Descaling is needed every few weeks with hard water, manual explains the process clearly."]},
    {"id": "p010", "name": "NightGlow Desk Lamp", "category": "furniture", "price": 1799, "rating": 4.3,
     "specs": {"brightness_levels": 5, "color_temp": "Adjustable 3000K-6000K", "power": "USB-C"},
     "reviews": [
         "Warm light setting is great for evening reading, doesn't strain the eyes.",
         "USB-C charging port on the base is a nice touch for charging a phone.",
         "Touch controls are responsive and easy to use in the dark.",
         "Arm doesn't hold position perfectly at the highest extension, tends to droop slightly."]},
    {"id": "p011", "name": "SwiftCharge 65W GaN Charger", "category": "accessories", "price": 1999, "rating": 4.4,
     "specs": {"output": "65W", "ports": "2x USB-C, 1x USB-A", "size": "Compact GaN"},
     "reviews": [
         "Charges a 14 inch laptop and phone simultaneously without slowing down noticeably.",
         "Much smaller than the stock charger it replaced, easy to travel with.",
         "Gets warm under full load but nothing concerning.",
         "Works fine with third-party USB-C cables, no compatibility issues so far."]},
    {"id": "p012", "name": "CloudRest Memory Foam Pillow", "category": "home", "price": 1299, "rating": 3.8,
     "specs": {"material": "Memory foam", "cover": "Removable, washable", "firmness": "Medium"},
     "reviews": [
         "Took about a week to stop noticing the new foam smell.",
         "Good neck support for side sleepers, less ideal if you sleep on your stomach.",
         "Cover is easy to remove and wash, dries fast.",
         "Firmness softened slightly after a month of use, still supportive though."]},
]
CATALOG_BY_ID = {p["id"]: p for p in CATALOG}
CATEGORY_WORDS = {"laptop", "audio", "monitor", "furniture", "kitchen", "accessories", "home"}


# =============================================================================
# CATALOG SERVICE — deterministic search/details/compare
# =============================================================================

def catalog_search(query: str, max_price: Optional[float] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
    q = (query or "").lower().strip()
    terms = [t for t in q.split() if t]
    results = []
    for p in CATALOG:
        if max_price is not None and p["price"] > max_price:
            continue
        if category and category.lower() != p["category"].lower():
            continue
        haystack = f"{p['name']} {p['category']}".lower()
        if not terms or any(t in haystack for t in terms):
            results.append(p)
    if not results and (max_price is not None or category):
        for p in CATALOG:
            if max_price is not None and p["price"] > max_price:
                continue
            if category and category.lower() != p["category"].lower():
                continue
            results.append(p)
    results.sort(key=lambda p: (-p["rating"], p["price"]))
    return [_summary(p) for p in results[:8]]


def _summary(p: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": p["id"], "name": p["name"], "price": p["price"], "rating": p["rating"], "category": p["category"]}


def catalog_details(product_id: str) -> Optional[Dict[str, Any]]:
    p = CATALOG_BY_ID.get(product_id)
    if not p:
        return None
    return {"id": p["id"], "name": p["name"], "category": p["category"], "price": p["price"],
            "rating": p["rating"], "specs": p["specs"], "review_snippets": p["reviews"][:3]}


def catalog_compare(product_ids: List[str]) -> Dict[str, Any]:
    rows = []
    for pid in product_ids:
        p = CATALOG_BY_ID.get(pid)
        if p:
            rows.append({"id": p["id"], "name": p["name"], "price": p["price"], "rating": p["rating"], "specs": p["specs"]})
    return {"products": rows}


# =============================================================================
# RAG REVIEW ENGINE — TF-IDF retrieval, grounded-or-nothing
# =============================================================================

def rag_ask(product_id: str, question: str) -> Dict[str, Any]:
    product = CATALOG_BY_ID.get(product_id)
    reviews = product["reviews"] if product else []
    if not reviews or not product:
        return {"grounded": False, "answer": "No reviews available for this product.", "citations": []}

    corpus = reviews + [question]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return {"grounded": False, "answer": "I couldn't find review content relevant to that question.", "citations": []}

    question_vec = matrix[-1]
    review_vecs = matrix[:-1]
    sims = cosine_similarity(question_vec, review_vecs)[0]
    ranked = sorted(zip(reviews, sims), key=lambda x: -x[1])
    top = [r for r, score in ranked if score >= GROUNDING_THRESHOLD][:2]

    if not top:
        return {"grounded": False,
                "answer": "I don't have a review-grounded answer to that for this product — the review corpus doesn't cover it.",
                "citations": []}
    return {"grounded": True, "answer": f"Based on reviews for {product['name']}: " + " ".join(top), "citations": top}


# =============================================================================
# RAZORPAY CLIENT — mocked test mode by default, live client optional
# =============================================================================

class RazorpayMockClient:
    def create_order(self, amount: float, currency: str = "INR") -> Dict[str, Any]:
        return {"id": f"order_mock_{uuid.uuid4().hex[:14]}", "amount": amount, "currency": currency, "status": "created"}

    def capture_payment(self, order_id: str, amount: float) -> Dict[str, Any]:
        time.sleep(0.15)
        if random.random() < SIMULATED_PAYMENT_FAILURE_RATE:
            return {"status": "failed", "order_id": order_id, "error": "simulated_gateway_timeout"}
        return {"status": "captured", "order_id": order_id, "payment_id": f"pay_mock_{uuid.uuid4().hex[:14]}", "amount": amount}


class RazorpayLiveClient:
    def __init__(self, key_id: str, key_secret: str):
        import razorpay  # lazy import — never required unless RAZORPAY_LIVE=true
        self.client = razorpay.Client(auth=(key_id, key_secret))

    def create_order(self, amount: float, currency: str = "INR") -> Dict[str, Any]:
        order = self.client.order.create({"amount": int(amount * 100), "currency": currency, "payment_capture": 1})
        return {"id": order["id"], "amount": amount, "currency": currency, "status": order["status"]}

    def capture_payment(self, order_id: str, amount: float) -> Dict[str, Any]:
        # A real conversational-checkout flow needs Razorpay Checkout on the
        # client + signature verification here — documented extension point.
        raise NotImplementedError("Live payment capture requires client-side Razorpay Checkout + signature verification.")


def _get_razorpay_client():
    if RAZORPAY_LIVE and RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
        return RazorpayLiveClient(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)
    return RazorpayMockClient()


razorpay_client = _get_razorpay_client()


# =============================================================================
# AUDIT LOG — every tool call + transaction, appended with a timestamp
# =============================================================================

def audit_log(session_id: str, tool: str, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> None:
    entry = {"ts": time.time(), "session_id": session_id, "tool": tool, "input": input_data, "output": output_data}
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def audit_read_all() -> List[Dict[str, Any]]:
    try:
        with open(AUDIT_LOG_PATH, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


# =============================================================================
# CART / SESSION STATE — in-memory
# =============================================================================

_carts: Dict[str, Dict[str, int]] = {}
_pending_confirmation: Dict[str, bool] = {}


def cart_add(session_id: str, product_id: str, qty: int = 1) -> Dict[str, Any]:
    cart = _carts.setdefault(session_id, {})
    cart[product_id] = cart.get(product_id, 0) + qty
    _pending_confirmation[session_id] = False
    return cart_view(session_id)


def cart_view(session_id: str) -> Dict[str, Any]:
    cart = _carts.get(session_id, {})
    items, total = [], 0.0
    for pid, qty in cart.items():
        p = CATALOG_BY_ID.get(pid)
        if not p:
            continue
        line_total = p["price"] * qty
        total += line_total
        items.append({"product_id": pid, "name": p["name"], "qty": qty, "unit_price": p["price"], "line_total": line_total})
    return {"items": items, "total": total}


def cart_clear(session_id: str) -> None:
    _carts[session_id] = {}
    _pending_confirmation[session_id] = False


def is_pending_confirmation(session_id: str) -> bool:
    return _pending_confirmation.get(session_id, False)


def set_pending_confirmation(session_id: str, pending: bool) -> None:
    _pending_confirmation[session_id] = pending


# =============================================================================
# TOOLS — the 6 agent-facing tools, matching the MCP schema from the deck.
# Every call is audit-logged. checkout() is bounded + gated in code.
# =============================================================================

def tool_search_products(session_id: str, query: str, max_price: float = None, category: str = None) -> Dict[str, Any]:
    result = {"products": catalog_search(query, max_price, category)}
    audit_log(session_id, "search_products", {"query": query, "max_price": max_price, "category": category}, result)
    return result


def tool_get_product_details(session_id: str, product_id: str) -> Dict[str, Any]:
    result = catalog_details(product_id) or {"error": "product_not_found"}
    audit_log(session_id, "get_product_details", {"product_id": product_id}, result)
    return result


def tool_compare_products(session_id: str, product_ids: List[str]) -> Dict[str, Any]:
    result = catalog_compare(product_ids)
    audit_log(session_id, "compare_products", {"product_ids": product_ids}, result)
    return result


def tool_ask_reviews(session_id: str, product_id: str, question: str) -> Dict[str, Any]:
    result = rag_ask(product_id, question)
    audit_log(session_id, "ask_reviews", {"product_id": product_id, "question": question}, result)
    return result


def tool_add_to_cart(session_id: str, product_id: str, qty: int = 1) -> Dict[str, Any]:
    result = {"error": "product_not_found"} if not CATALOG_BY_ID.get(product_id) else cart_add(session_id, product_id, qty)
    audit_log(session_id, "add_to_cart", {"product_id": product_id, "qty": qty}, result)
    return result


def tool_checkout(session_id: str, confirmed: bool = False) -> Dict[str, Any]:
    cart = cart_view(session_id)

    if not cart["items"]:
        result = {"status": "rejected", "reason": "cart_empty"}
        audit_log(session_id, "checkout", {"confirmed": confirmed}, result)
        return result

    if not confirmed:
        set_pending_confirmation(session_id, True)
        result = {"status": "confirmation_required", "cart": cart}
        audit_log(session_id, "checkout", {"confirmed": confirmed}, result)
        return result

    if cart["total"] > MAX_CHECKOUT_AMOUNT:
        result = {"status": "rejected", "reason": "exceeds_spend_cap", "cap": MAX_CHECKOUT_AMOUNT, "total": cart["total"]}
        audit_log(session_id, "checkout", {"confirmed": confirmed}, result)
        return result

    order = razorpay_client.create_order(cart["total"])
    payment = razorpay_client.capture_payment(order["id"], cart["total"])

    if payment["status"] != "captured":
        result = {"status": "failed", "reason": payment.get("error", "payment_failed"), "order_id": order["id"], "cart": cart}
        audit_log(session_id, "checkout", {"confirmed": confirmed}, result)
        return result

    cart_clear(session_id)
    result = {"status": "success", "order_id": order["id"], "payment_id": payment["payment_id"], "amount": payment["amount"]}
    audit_log(session_id, "checkout", {"confirmed": confirmed}, result)
    return result


# =============================================================================
# RULE-BASED AGENT — default, always-works path. No LLM, no external calls.
# =============================================================================

def _extract_max_price(message: str) -> Optional[float]:
    m = re.search(r"under\s+(?:rs\.?|inr|₹)?\s*([\d,]+)", message, re.I) or \
        re.search(r"(?:below|less than)\s+(?:rs\.?|inr|₹)?\s*([\d,]+)", message, re.I)
    return float(m.group(1).replace(",", "")) if m else None


def _extract_category(message: str) -> Optional[str]:
    lower = message.lower()
    for cat in CATEGORY_WORDS:
        if cat in lower:
            return cat
    return None


def _find_mentioned_product(message: str) -> Optional[Dict[str, Any]]:
    lower = message.lower()
    best = None
    for p in CATALOG:
        name_words = p["name"].lower().split()
        hits = sum(1 for w in name_words if w in lower and len(w) > 2)
        if hits >= 2 and (best is None or hits > best[1]):
            best = (p, hits)
    return best[0] if best else None


def _format_checkout_result(result: Dict[str, Any]) -> Dict[str, Any]:
    status = result.get("status")
    if status == "confirmation_required":
        cart = result["cart"]
        lines = [f"{i['name']} x{i['qty']} — ₹{i['line_total']:.0f}" for i in cart["items"]]
        return {"reply": "Here's your order:\n" + "\n".join(lines) +
                          f"\nTotal: ₹{cart['total']:.0f}\n\nConfirm checkout via Razorpay test-mode? (yes/no)",
                "cart": cart, "pending_confirmation": True}
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


def rule_based_handle(session_id: str, message: str) -> Dict[str, Any]:
    lower = message.lower().strip()

    if is_pending_confirmation(session_id):
        if lower in ("yes", "confirm", "y", "confirm checkout", "proceed"):
            return _format_checkout_result(tool_checkout(session_id, confirmed=True))
        if lower in ("no", "cancel", "n"):
            set_pending_confirmation(session_id, False)
            return {"reply": "Checkout cancelled — your cart is still saved.", "cart": cart_view(session_id)}

    if any(k in lower for k in ["checkout", "buy now", "place order", "complete purchase"]):
        return _format_checkout_result(tool_checkout(session_id, confirmed=False))

    if "cart" in lower and any(k in lower for k in ["add", "put"]):
        product = _find_mentioned_product(message)
        if not product:
            return {"reply": "Which product would you like to add? Try naming it, e.g. \"add AeroBook 14 to cart\"."}
        qty_match = re.search(r"(?:qty|quantity|x)\s*[:=]?\s*(\d+)\b", lower) or \
                    re.search(r"\b(\d+)\s*(?:units?|pieces?|pcs)\b", lower)
        qty = int(qty_match.group(1)) if qty_match else 1
        result = tool_add_to_cart(session_id, product["id"], qty)
        return {"reply": f"Added {qty} x {product['name']} to your cart. Current total: ₹{result['total']:.0f}.", "cart": result}

    if any(k in lower for k in ["review", "reviews say", "people say", "worth it", "battery life", "is it good"]):
        product = _find_mentioned_product(message)
        if not product:
            return {"reply": "Which product's reviews should I check? Name the product and your question."}
        result = tool_ask_reviews(session_id, product["id"], message)
        prefix = "" if result["grounded"] else "(Not grounded in reviews) "
        return {"reply": prefix + result["answer"]}

    if "compare" in lower or " vs " in lower or " versus " in lower:
        mentioned = [p for p in CATALOG if p["name"].lower() in lower]
        if len(mentioned) < 2:
            candidates = catalog_search(message)
            ids = [c["id"] for c in candidates[:3]]
        else:
            ids = [p["id"] for p in mentioned]
        result = tool_compare_products(session_id, ids)
        if not result["products"]:
            return {"reply": "I couldn't find matching products to compare. Try naming two products directly."}
        lines = [f"{p['name']} — ₹{p['price']:.0f}, rating {p['rating']}" for p in result["products"]]
        return {"reply": "Here's the comparison:\n" + "\n".join(lines), "products": result["products"]}

    max_price = _extract_max_price(message)
    category = _extract_category(message)
    result = tool_search_products(session_id, message, max_price, category)
    products = result["products"]
    if not products:
        return {"reply": "I couldn't find anything matching that. Try a different keyword or a higher budget."}
    lines = [f"{p['name']} — ₹{p['price']:.0f} (rating {p['rating']})" for p in products[:5]]
    return {"reply": "Here's what I found:\n" + "\n".join(lines), "products": products}


# =============================================================================
# OPTIONAL LLM AGENT — Claude with native tool use. Only used if
# USE_LLM_AGENT=true and ANTHROPIC_API_KEY is set. Falls back silently
# to the rule-based agent on any error.
# =============================================================================

_LLM_TOOL_SPECS = [
    {"name": "search_products", "description": "Search the catalog by keyword, optional max price, optional category.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"}, "max_price": {"type": "number"}, "category": {"type": "string"}},
         "required": ["query"]}},
    {"name": "get_product_details", "description": "Get full specs and review snippets for one product by id.",
     "input_schema": {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]}},
    {"name": "compare_products", "description": "Compare multiple products side by side by their ids.",
     "input_schema": {"type": "object", "properties": {
         "product_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["product_ids"]}},
    {"name": "ask_reviews", "description": "Ask a review-grounded question about one product. Returns grounded=false if unsupported.",
     "input_schema": {"type": "object", "properties": {
         "product_id": {"type": "string"}, "question": {"type": "string"}}, "required": ["product_id", "question"]}},
    {"name": "add_to_cart", "description": "Add a product to the session's cart.",
     "input_schema": {"type": "object", "properties": {
         "product_id": {"type": "string"}, "qty": {"type": "integer"}}, "required": ["product_id"]}},
    {"name": "checkout", "description": "Checkout the current cart via Razorpay test mode. Call with confirmed=false "
                                         "first to show the order summary; only call with confirmed=true after the "
                                         "user explicitly agrees.",
     "input_schema": {"type": "object", "properties": {"confirmed": {"type": "boolean"}}}},
]

_SYSTEM_PROMPT = """You are CommerceCopilot, a shopping assistant for a merchant catalog.
Only state facts returned by tool calls. Never invent prices, specs, or review content.
For any question about reviews, call ask_reviews rather than guessing.
Never call checkout with confirmed=true unless the user has explicitly confirmed an order
you already showed them. Keep replies short and concrete."""


def _dispatch_llm_tool(session_id: str, name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    if name == "search_products":
        return tool_search_products(session_id, tool_input.get("query", ""), tool_input.get("max_price"), tool_input.get("category"))
    if name == "get_product_details":
        return tool_get_product_details(session_id, tool_input["product_id"])
    if name == "compare_products":
        return tool_compare_products(session_id, tool_input.get("product_ids", []))
    if name == "ask_reviews":
        return tool_ask_reviews(session_id, tool_input["product_id"], tool_input["question"])
    if name == "add_to_cart":
        return tool_add_to_cart(session_id, tool_input["product_id"], tool_input.get("qty", 1))
    if name == "checkout":
        return tool_checkout(session_id, tool_input.get("confirmed", False))
    return {"error": f"unknown_tool:{name}"}


def llm_handle(session_id: str, message: str, max_steps: int = 4) -> Dict[str, Any]:
    import anthropic  # lazy import — never required unless USE_LLM_AGENT=true
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": message}]

    for _ in range(max_steps):
        response = client.messages.create(
            model=LLM_MODEL, max_tokens=1024, system=_SYSTEM_PROMPT,
            tools=_LLM_TOOL_SPECS, messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            text = "".join(b.text for b in response.content if b.type == "text")
            return {"reply": text}

        tool_results = []
        for block in tool_uses:
            result = _dispatch_llm_tool(session_id, block.name, block.input)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})

    return {"reply": "I wasn't able to finish that within my step limit — could you rephrase or simplify the request?"}


def handle_message(session_id: str, message: str) -> Dict[str, Any]:
    if USE_LLM_AGENT and ANTHROPIC_API_KEY:
        try:
            result = llm_handle(session_id, message)
            return {"reply": result["reply"], "cart": cart_view(session_id),
                    "pending_confirmation": is_pending_confirmation(session_id)}
        except Exception as e:
            fallback = rule_based_handle(session_id, message)
            fallback["reply"] = f"[LLM agent unavailable ({type(e).__name__}), used fallback] " + fallback["reply"]
            return fallback
    return rule_based_handle(session_id, message)


# =============================================================================
# FASTAPI APP
# =============================================================================

class ChatRequest(BaseModel):
    session_id: str
    message: str


app = FastAPI(title="CommerceCopilot (single-file)")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    result = handle_message(req.session_id, req.message)
    return {
        "reply": result.get("reply", ""),
        "products": result.get("products"),
        "cart": result.get("cart") or cart_view(req.session_id),
        "pending_confirmation": result.get("pending_confirmation", is_pending_confirmation(req.session_id)),
    }


@app.get("/api/tools/audit_log")
def get_audit_log():
    return audit_read_all()


# =============================================================================
# EMBEDDED FRONTEND — plain HTML/CSS/JS, no build step
# =============================================================================

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CommerceCopilot</title>
<style>
:root { --navy:#1e2761; --teal:#0f6e56; --teal-light:#e1f5ee; --border:#e3e6f0; --bg-card:#f7f8fc; --text-muted:#5f5e5a; }
* { box-sizing: border-box; }
body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; color:var(--navy); }
.app-shell { display:grid; grid-template-columns:1fr 320px; height:100vh; }
.chat-pane { display:flex; flex-direction:column; border-right:1px solid var(--border); }
.chat-header { padding:16px 24px; border-bottom:1px solid var(--border); }
.chat-header h1 { margin:0; font-size:18px; }
.chat-header p { margin:2px 0 0; font-size:12px; color:var(--text-muted); }
.messages { flex:1; overflow-y:auto; padding:20px 24px; display:flex; flex-direction:column; gap:14px; }
.msg-row { display:flex; }
.msg-row.user { justify-content:flex-end; }
.bubble { max-width:70%; padding:10px 14px; border-radius:12px; font-size:14px; line-height:1.5; white-space:pre-wrap; }
.msg-row.user .bubble { background:var(--navy); color:#fff; }
.msg-row.assistant .bubble { background:var(--bg-card); border:1px solid var(--border); }
.product-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr)); gap:10px; margin-top:8px; max-width:70%; }
.product-card { border:1px solid var(--border); border-radius:10px; padding:10px; background:#fff; }
.product-card .name { font-size:13px; font-weight:600; margin-bottom:4px; }
.product-card .meta { font-size:12px; color:var(--text-muted); }
.confirm-bar { display:flex; gap:8px; padding:0 24px 16px; }
.confirm-bar button { padding:8px 14px; border-radius:8px; border:1px solid var(--border); font-size:13px; cursor:pointer; background:#fff; }
.confirm-bar button.confirm { background:var(--teal); color:#fff; border-color:var(--teal); }
.composer { display:flex; gap:8px; padding:16px 24px; border-top:1px solid var(--border); }
.composer input { flex:1; padding:10px 14px; border-radius:8px; border:1px solid var(--border); font-size:14px; }
.composer button { padding:10px 18px; border-radius:8px; border:none; background:var(--navy); color:#fff; font-size:14px; cursor:pointer; }
.composer button:disabled { opacity:.5; cursor:default; }
.cart-pane { padding:20px; overflow-y:auto; }
.cart-pane h2 { font-size:15px; margin:0 0 12px; }
.cart-item { display:flex; justify-content:space-between; font-size:13px; padding:8px 0; border-bottom:1px solid var(--border); }
.cart-item .qty { color:var(--text-muted); }
.cart-total { display:flex; justify-content:space-between; font-weight:600; padding-top:12px; font-size:14px; }
.empty-cart { color:var(--text-muted); font-size:13px; }
</style>
</head>
<body>
<div class="app-shell">
  <div class="chat-pane">
    <div class="chat-header">
      <h1>CommerceCopilot</h1>
      <p>Agentic checkout assistant · Razorpay test mode</p>
    </div>
    <div class="messages" id="messages"></div>
    <div class="confirm-bar" id="confirmBar" style="display:none">
      <button class="confirm" onclick="send('yes')">Confirm checkout</button>
      <button onclick="send('no')">Cancel</button>
    </div>
    <div class="composer">
      <input id="input" placeholder="Ask about products, reviews, or say checkout…" onkeydown="if(event.key==='Enter')send()"/>
      <button id="sendBtn" onclick="send()">Send</button>
    </div>
  </div>
  <div class="cart-pane">
    <h2>Cart</h2>
    <div id="cart"><div class="empty-cart">Your cart is empty.</div></div>
  </div>
</div>
<script>
const sessionId = "session_" + Math.random().toString(36).slice(2, 10);
const messagesEl = document.getElementById("messages");
const cartEl = document.getElementById("cart");
const confirmBar = document.getElementById("confirmBar");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");

addBubble("assistant", "Hi! I can search the catalog, compare products, answer review-grounded questions, and check out via Razorpay test mode. Try: \\"find me a laptop under 50000\\".");

function addBubble(role, text, products) {
  const row = document.createElement("div");
  row.className = "msg-row " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  row.appendChild(bubble);
  if (products && products.length) {
    const grid = document.createElement("div");
    grid.className = "product-grid";
    products.forEach(p => {
      const card = document.createElement("div");
      card.className = "product-card";
      card.innerHTML = `<div class="name">${p.name}</div><div class="meta">₹${p.price.toLocaleString("en-IN")} · ★ ${p.rating}</div>`;
      grid.appendChild(card);
    });
    row.appendChild(grid);
  }
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderCart(cart) {
  if (!cart || !cart.items || cart.items.length === 0) {
    cartEl.innerHTML = '<div class="empty-cart">Your cart is empty.</div>';
    return;
  }
  let html = "";
  cart.items.forEach(item => {
    html += `<div class="cart-item"><span>${item.name} <span class="qty">x${item.qty}</span></span><span>₹${item.line_total.toLocaleString("en-IN")}</span></div>`;
  });
  html += `<div class="cart-total"><span>Total</span><span>₹${cart.total.toLocaleString("en-IN")}</span></div>`;
  cartEl.innerHTML = html;
}

async function send(forcedText) {
  const text = forcedText !== undefined ? forcedText : inputEl.value;
  if (!text.trim()) return;
  addBubble("user", text);
  inputEl.value = "";
  sendBtn.disabled = true;
  try {
    const res = await fetch("/chat", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({session_id: sessionId, message: text}),
    });
    const data = await res.json();
    addBubble("assistant", data.reply, data.products);
    renderCart(data.cart);
    confirmBar.style.display = data.pending_confirmation ? "flex" : "none";
  } catch (e) {
    addBubble("assistant", "Couldn't reach the backend — is it still running?");
  } finally {
    sendBtn.disabled = false;
  }
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
