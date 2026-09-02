from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat, tools_router
from app.config import CORS_ORIGINS

app = FastAPI(title="CommerceCopilot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(tools_router.router)


@app.get("/health")
def health():
    return {"status": "ok"}
