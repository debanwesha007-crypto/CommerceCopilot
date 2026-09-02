from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    products: Optional[List[Dict[str, Any]]] = None
    cart: Optional[Dict[str, Any]] = None
    pending_confirmation: bool = False
    tool_calls: Optional[List[Dict[str, Any]]] = None


class SearchRequest(BaseModel):
    query: str
    max_price: Optional[float] = None
    category: Optional[str] = None


class ProductDetailsRequest(BaseModel):
    product_id: str


class CompareRequest(BaseModel):
    product_ids: List[str]


class AskReviewsRequest(BaseModel):
    product_id: str
    question: str


class AddToCartRequest(BaseModel):
    session_id: str
    product_id: str
    qty: int = 1


class CheckoutRequest(BaseModel):
    session_id: str
