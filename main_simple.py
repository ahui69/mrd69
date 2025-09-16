#!/usr/bin/env python3
"""
Prosty serwer FastAPI - wersja minimalna do testów
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Serwuj pliki statyczne
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            # Prosty echo - odbij wiadomość z prefiksem
            response = f"Echo: {message}"
            await websocket.send_text(response)
    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    import uvicorn

    try:
        import config  # użyj wspólnej konfiguracji jeśli dostępna

        host = getattr(config, "HOST", "0.0.0.0")
        port = int(getattr(config, "PORT", 5959))
    except Exception:
        host, port = "0.0.0.0", 5959
    uvicorn.run(app, host=host, port=port)
