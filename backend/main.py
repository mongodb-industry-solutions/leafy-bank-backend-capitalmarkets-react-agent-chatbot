import logging
from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware

# Add market assistant API router
from api_market_assistant import router as market_assistant_router
# Add checkpointer API router
from api_checkpointer import router as checkpointer_router

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

@app.get("/")
async def read_root(request: Request):
    return {"message": "Server is running"}

# Add the market assistant router to the main app
app.include_router(market_assistant_router)
# Add the checkpointer router to the main app
app.include_router(checkpointer_router)