"""
Web UI for the AI Assistant using FastAPI.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="AI Assistant", description="Advanced AI Assistant with LangChain and ChromaDB")

# Set up templates directory
templates_dir = Path(__file__).parent.parent.parent / "src" / "utils" / "templates"
if not templates_dir.exists():
    templates_dir.mkdir(parents=True)
    
    # Create a basic HTML template if it doesn't exist
    index_html = templates_dir / "index.html"
    if not index_html.exists():
        with open(index_html, "w") as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>AI Assistant</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .chat-container {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            height: 70vh;
            display: flex;
            flex-direction: column;
        }
        .messages {
            flex-grow: 1;
            overflow-y: auto;
            margin-bottom: 20px;
            padding: 10px;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
        }
        .user {
            background-color: #e3f2fd;
            margin-left: 20px;
            margin-right: 5px;
        }
        .assistant {
            background-color: #f1f8e9;
            margin-right: 20px;
            margin-left: 5px;
        }
        .input-container {
            display: flex;
        }
        #messageInput {
            flex-grow: 1;
            padding: 10px;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            margin-right: 10px;
        }
        button {
            padding: 10px 20px;
            background-color: #4caf50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #388e3c;
        }
        .status {
            text-align: center;
            margin-top: 10px;
            color: #757575;
        }
    </style>
</head>
<body>
    <h1>AI Assistant</h1>
    <div class="chat-container">
        <div class="messages" id="messages"></div>
        <div class="input-container">
            <input type="text" id="messageInput" placeholder="Type your message here..." />
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
    <div class="status" id="status">Disconnected</div>

    <script>
        let socket;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        
        function connectWebSocket() {
            const statusElement = document.getElementById('status');
            statusElement.textContent = 'Connecting...';
            
            // Create WebSocket connection
            socket = new WebSocket(`ws://${window.location.host}/ws`);
            
            socket.onopen = function(e) {
                console.log('WebSocket connection established');
                statusElement.textContent = 'Connected';
                reconnectAttempts = 0;
            };
            
            socket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                addMessage(data.sender, data.message);
            };
            
            socket.onclose = function(event) {
                console.log('WebSocket connection closed');
                statusElement.textContent = 'Disconnected';
                
                // Attempt to reconnect
                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    statusElement.textContent = `Reconnecting (${reconnectAttempts}/${maxReconnectAttempts})...`;
                    setTimeout(connectWebSocket, 2000);
                } else {
                    statusElement.textContent = 'Failed to connect. Please refresh the page.';
                }
            };
            
            socket.onerror = function(error) {
                console.error('WebSocket error:', error);
                statusElement.textContent = 'Connection error';
            };
        }
        
        function sendMessage() {
            const messageInput = document.getElementById('messageInput');
            const message = messageInput.value.trim();
            
            if (message && socket && socket.readyState === WebSocket.OPEN) {
                // Add user message to chat
                addMessage('user', message);
                
                // Send message to server
                socket.send(JSON.stringify({ message: message }));
                
                // Clear input
                messageInput.value = '';
            }
        }
        
        function addMessage(sender, message) {
            const messagesContainer = document.getElementById('messages');
            const messageElement = document.createElement('div');
            messageElement.classList.add('message', sender);
            messageElement.textContent = message;
            messagesContainer.appendChild(messageElement);
            
            // Scroll to bottom
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        // Handle Enter key press
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Connect when page loads
        window.onload = connectWebSocket;
    </script>
</body>
</html>
            """)

templates = Jinja2Templates(directory=str(templates_dir))

# Global variable to store the assistant instance
_assistant = None


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_json({
            "sender": "assistant",
            "message": message
        })


# Create connection manager
manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for chat."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")
            
            if not user_message:
                continue
            
            logger.info(f"Received message: {user_message}")
            
            # Process message with assistant
            if _assistant is not None:
                try:
                    response = _assistant.chat(user_message)
                    await manager.send_message(response, websocket)
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    await manager.send_message(f"Error: {str(e)}", websocket)
            else:
                await manager.send_message("Assistant not initialized properly.", websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)


def start_web_ui(assistant: Any, host: str = "127.0.0.1", port: int = 8000) -> None:
    """
    Start the web UI.
    
    Args:
        assistant: The assistant instance
        host: Host to bind the server to
        port: Port to bind the server to
    """
    global _assistant
    _assistant = assistant
    
    logger.info(f"Starting web UI on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port) 