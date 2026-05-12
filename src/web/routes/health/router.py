from fastapi import APIRouter
from fastapi.responses import ORJSONResponse

router = APIRouter()

@router.get("/health")
def health():
    return ORJSONResponse(content={"status": "ok"})