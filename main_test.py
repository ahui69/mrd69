import time

from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok", "ts": time.time()}


@app.get("/")
async def root():
    return {"message": "Test server"}
