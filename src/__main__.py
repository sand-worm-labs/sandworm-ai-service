from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, RedirectResponse

from src.web.routes.chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    yield
    # shutdown


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
    return ORJSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: Exception):
    return ORJSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


app.include_router(chat_router)


@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok"}