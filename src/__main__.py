from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from src.config.settings import settings
from src.util.redis_client import init_redis, close_redis
from src.util.qdrant import init_qdrant, close_qdrant
from src.web.middleware.auth import verify_handshake
from src.web.routes.health.router import router as health_router
from src.web.routes.chat.title import router as chat_title_router
from src.web.routes.chat.completions import router as completions_router
from src.web.routes.document.title import router as document_title_router
from src.web.routes.intent.test_intent import router as intent_router
from src.web.routes.code.router import router as code_router
from src.web.routes.sql.router import router as sql_router
from src.web.routes.markdown.router import router as markdown_router
from src.web.routes.select_tool.router import router as select_tool_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_redis(settings.redis_url)
    # await init_qdrant(settings.qdrant_url, settings.qdrant_api_key)
    yield
    await close_redis()
    await close_qdrant()

app = FastAPI(
    title="Sandworm AI Service",
    lifespan=lifespan,
    redoc_url=None,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def exception_handler(_, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})



app.include_router(
    completions_router,
    prefix="/chat",
    dependencies=[Depends(verify_handshake)],
)

app.include_router(
    chat_title_router,
    prefix="/chat",
    dependencies=[Depends(verify_handshake)],
)

app.include_router(
    document_title_router,
    prefix="/document",
    dependencies=[Depends(verify_handshake)],
)
app.include_router(
    intent_router,
    prefix="/intent",
    dependencies=[Depends(verify_handshake)],
)

app.include_router(
    code_router,
    prefix="/code",
    dependencies=[Depends(verify_handshake)],
)

app.include_router(
    sql_router,
    prefix="/sql",
    dependencies=[Depends(verify_handshake)],
)

app.include_router(
    markdown_router,
    prefix="/markdown",
    dependencies=[Depends(verify_handshake)],
)

# app.include_router(
#     select_tool_router,
#     prefix="/select-tool",
#     dependencies=[Depends(verify_handshake)],
# )

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

app.include_router(health_router)
