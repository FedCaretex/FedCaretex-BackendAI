from fastapi import APIRouter, HTTPException

from app.schemas import AuthResponse, LoginRequest, RegisterRequest
from app.store import store

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
def register(body: RegisterRequest):
    try:
        user = store.register(body.name, body.email, body.password)
        session = store.login(body.email, body.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AuthResponse(
        access_token=session["access_token"],
        user_id=user["user_id"],
        name=user["name"],
        email=user["email"],
    )


@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest):
    try:
        session = store.login(body.email, body.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return AuthResponse(
        access_token=session["access_token"],
        user_id=session["user_id"],
        name=session["name"],
        email=session["email"],
    )
