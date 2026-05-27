from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, RedirectResponse
from src.web.middleware.auth import verify_handshake
from src.web.routes.health.router import router as health_router
from src.web.routes.chat.title import router as title_router
from src.web.routes.chat.completions import router as completions_router
from src.web.routes.document.title import router as document_router
from src.web.routes.intent.test_intent import router as intent_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(
    title="Sandworm AI Service",
    lifespan=lifespan,
    redoc_url=None,
    default_response_class=ORJSONResponse,
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
    return ORJSONResponse(status_code=500, content={"detail": str(exc)})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: RequestValidationError):
    return ORJSONResponse(status_code=400, content={"detail": str(exc)})



app.include_router(
    completions_router,
    prefix="/chat",
    dependencies=[Depends(verify_handshake)],
)
app.include_router(
    title_router,
    prefix="/chat",
    dependencies=[Depends(verify_handshake)],
)
app.include_router(
    document_router,
    prefix="/document",
    dependencies=[Depends(verify_handshake)],
)

app.include_router(
    intent_router,
    prefix="/intent",
    dependencies=[Depends(verify_handshake)],
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

app.include_router(health_router)
