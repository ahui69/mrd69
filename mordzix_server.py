"""
MORDZIX PRODUCTION SERVER - Production Ready FastAPI Server
===========================================================
"""

import json
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import Mordzix Core
from mordzix_core import mordzix_engine

app = FastAPI(title="Mordzix Production")

# CORS - Safari & Mobile Optimized
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "X-User-Agent",
        "User-Agent",
        "Cache-Control",
        "Pragma",
        "X-WebSocket-Extensions",
        "X-WebSocket-Protocol",
        "X-WebSocket-Version",
        "Sec-WebSocket-Key",
        "Sec-WebSocket-Version",
        "Sec-WebSocket-Extensions",
        "Connection",
        "Upgrade",
    ],
    expose_headers=["*"],
    max_age=86400,  # 24h preflight cache
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Models
class MordzixChatRequest(BaseModel):
    thread_id: str | None = None
    user_id: str
    content: str
    message_type: str = "text"
    attachments: list[dict[str, Any]] | None = None


class MordzixThreadRequest(BaseModel):
    user_id: str
    title: str = "New Chat"


# Basic endpoints
@app.get("/")
async def root():
    return {"message": "🚀 Mordzix API - Ziomek bez kagańca!", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "ok", "message": "Mordzix działa jak bestia! 💪"}


@app.get("/mordzix")
async def mordzix_interface():
    """Główny interfejs Mordzix - production chat."""
    return FileResponse("static/mordzix_safari.html")


# Mordzix API endpoints
@app.post("/mordzix/chat")
async def mordzix_chat(request: MordzixChatRequest):
    """Główny endpoint czatu Mordzix - bez kagańca, bez filtrów!"""
    try:
        # Create thread if doesn't exist
        if not request.thread_id:
            thread = mordzix_engine.create_thread(request.user_id, "New Chat")
            thread_id = thread.id
        else:
            thread_id = request.thread_id

        # Process message through Mordzix engine
        ai_response = await mordzix_engine.process_message(
            thread_id, request.user_id, request.content, request.message_type
        )

        return {
            "status": "success",
            "thread_id": thread_id,
            "message_id": ai_response.id,
            "content": ai_response.content,
            "timestamp": ai_response.timestamp,
            "message_type": ai_response.message_type,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"status": "error", "message": f"Mordzix error: {str(e)}"}
        )


@app.post("/mordzix/thread/create")
async def create_mordzix_thread(request: MordzixThreadRequest):
    """Tworzy nowy wątek rozmowy z Mordzix."""
    try:
        thread = mordzix_engine.create_thread(request.user_id, request.title)
        return {
            "status": "success",
            "thread_id": thread.id,
            "title": thread.title,
            "created_at": thread.created_at,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/mordzix/thread/{thread_id}/history")
async def get_thread_history(thread_id: str, limit: int = 100):
    """Pobiera historię wiadomości z wątku."""
    try:
        if thread_id not in mordzix_engine.message_history:
            return JSONResponse(
                status_code=404, content={"status": "error", "message": "Thread not found"}
            )

        messages = mordzix_engine.message_history[thread_id][-limit:]
        return {
            "status": "success",
            "thread_id": thread_id,
            "messages": [
                {
                    "id": msg.id,
                    "user_id": msg.user_id,
                    "content": msg.content,
                    "message_type": msg.message_type,
                    "timestamp": msg.timestamp,
                    "status": msg.status,
                    "attachments": msg.attachments,
                }
                for msg in messages
            ],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.websocket("/mordzix/ws/{user_id}")
async def mordzix_websocket(websocket: WebSocket, user_id: str):
    """WebSocket endpoint dla real-time komunikacji z Mordzix."""
    await websocket.accept()
    mordzix_engine.websocket_connections[user_id] = websocket

    try:
        # Send welcome message
        await websocket.send_text(
            json.dumps(
                {
                    "type": "system",
                    "content": "Yo! Mordzix online - jestem gotowy do gadania! 🚀",
                    "timestamp": time.time(),
                }
            )
        )

        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)

                # Process through chat engine
                if message_data.get("type") == "chat":
                    ai_response = await mordzix_engine.process_message(
                        message_data.get("thread_id"),
                        user_id,
                        message_data.get("content", ""),
                        message_data.get("message_type", "text"),
                    )

                    # Send response back
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "chat_response",
                                "message_id": ai_response.id,
                                "content": ai_response.content,
                                "timestamp": ai_response.timestamp,
                            }
                        )
                    )

            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_text(
                    json.dumps(
                        {"type": "error", "content": f"Error: {str(e)}", "timestamp": time.time()}
                    )
                )

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Clean up connection
        if user_id in mordzix_engine.websocket_connections:
            del mordzix_engine.websocket_connections[user_id]


@app.post("/mordzix/crypto/score")
async def get_crypto_score_mordzix(token_id: str):
    """Pobiera score tokenu przez Mordzix system."""
    try:
        import crypto_advisor_full

        score_data = crypto_advisor_full.calculate_token_score(token_id)

        # Format response in Mordzix style
        mordzix_response = f"""🚀 Crypto score dla {token_id.upper()}:
💰 Cena: ${score_data['price']:,.2f}
📊 Score: {score_data['scores']['composite']}/100
⚡ Liquidity: {score_data['scores']['liquidity']}/100
🛡️ Trust: {score_data['scores']['trust']}/100
👥 Community: {score_data['scores']['community']}/100
⚠️ Risk level: {score_data['risk_level']}

{mordzix_engine.personality.enhance_response('Analiza gotowa!')}"""

        return {
            "status": "success",
            "token_id": token_id,
            "raw_data": score_data,
            "mordzix_response": mordzix_response,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mordzix_server:app", host="127.0.0.1", port=8080, reload=True)
