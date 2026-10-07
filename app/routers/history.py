from fastapi import APIRouter

from app.schemas import HistoryItem
from app.store import store

router = APIRouter(tags=["history"])


@router.get("/history/{user_id}", response_model=list[HistoryItem])
def get_history(user_id: str):
    return [HistoryItem(**item) for item in store.get_history(user_id)]
