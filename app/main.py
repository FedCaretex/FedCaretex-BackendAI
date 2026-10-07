from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import audit, auth, dataset, fl, history, predict
from app.services.fl_service import get_coordinator


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Bangun koordinator (load data, ledger, bootstrap FL) sebelum menerima request.
    get_coordinator()
    yield


app = FastAPI(
    title="SmartWealth Diabetes AI",
    description=(
        "Gateway Federated Learning (FedAvg/FedProx) + Differential Privacy + audit trail "
        "blockchain + SHAP eksak untuk deteksi dini risiko Diabetes Mellitus Tipe 2."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(fl.router)
app.include_router(audit.router)
app.include_router(history.router)
app.include_router(dataset.router)


@app.get("/")
def health():
    c = get_coordinator()
    return {
        "status": "ok",
        "service": "diabetes-risk-ai",
        "round_id": c.round,
        "ledger": c.ledger.kind,
        "data_source": c.data_source,
        "docs": "/docs",
    }
