from fastapi import FastAPI
from api.routes import anomalies, intel

app = FastAPI()

app.include_router(anomalies.router, prefix="/anomalies", tags=["anomalies"])
app.include_router(intel.router, prefix="/intel", tags=["intel"])