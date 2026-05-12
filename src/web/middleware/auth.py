from fastapi import Header, HTTPException
from src.config.settings import settings

async def verify_handshake(x_handshake_token: str = Header(...)):
    if x_handshake_token != settings.handshake_token:
        raise HTTPException(status_code=401, detail="Invalid handshake token")