import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
AUDIT_LOG_PATH = BASE_DIR / "audit_log.jsonl"

# --- LLM agent (optional) ---
# The app runs fully without these: the rule-based agent is the default,
# reliable path. Set USE_LLM_AGENT=true and provide ANTHROPIC_API_KEY to
# route chat through a LangChain tool-calling agent instead.
USE_LLM_AGENT = os.getenv("USE_LLM_AGENT", "false").lower() == "true"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

# --- Razorpay (test mode) ---
# Mocked by default so the demo runs with zero external dependencies.
# Set RAZORPAY_LIVE=true with real test-mode keys to hit the actual API.
RAZORPAY_LIVE = os.getenv("RAZORPAY_LIVE", "false").lower() == "true"
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")

# --- Guardrails ---
MAX_CHECKOUT_AMOUNT = int(os.getenv("MAX_CHECKOUT_AMOUNT", "100000"))  # paise-free unit cap per checkout
SIMULATED_PAYMENT_FAILURE_RATE = float(os.getenv("SIMULATED_PAYMENT_FAILURE_RATE", "0.1"))

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
