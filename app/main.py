from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.clients.vllm_client import load_model
from app.api.routes import reports, individual_reports,imageQuality,health


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Battery VLM Report API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(reports.router, prefix="/vlm")
app.include_router(individual_reports.router, prefix="/vlm")
app.include_router(imageQuality.router, prefix="/vlm")
