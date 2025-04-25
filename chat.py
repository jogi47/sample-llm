from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, Request, Depends, HTTPException, Body
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
from model import llm_handler
from config import Config

# Create router
router = APIRouter()

# Setup templates
templates = Jinja2Templates(directory="templates")

# Models
class ChatRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95

class ChatResponse(BaseModel):
    response: str
    model: str

class ModelChangeRequest(BaseModel):
    model: str

class ModelChangeResponse(BaseModel):
    success: bool
    message: str

# In-memory storage for chat
chat_history: Dict[str, List[Dict[str, Any]]] = {}

# Chat routes
@router.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    """Serve the chat interface page"""
    # Get available models from config
    available_models = Config.AVAILABLE_MODELS
    
    # Get current model
    current_model_key = os.environ.get("LLM_MODEL", Config.DEFAULT_MODEL)
    if current_model_key not in available_models:
        current_model_key = Config.DEFAULT_MODEL
    
    current_model_name = available_models[current_model_key]["name"]
    
    return templates.TemplateResponse(
        "chat.html", 
        {
            "request": request, 
            "available_models": available_models,
            "current_model_key": current_model_key,
            "current_model_name": current_model_name
        }
    )

@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate a response from the LLM model"""
    
    # Set model if specified
    if request.model and request.model in Config.AVAILABLE_MODELS:
        model_name = Config.AVAILABLE_MODELS[request.model]["name"]
        # Only change model if different from current
        if model_name != llm_handler.model_name:
            # Save current model
            original_model_name = llm_handler.model_name
            
            # Update model and reinitialize
            os.environ["LLM_MODEL"] = request.model
            # We need to reset the model handler to load the new model
            llm_handler.model_name = model_name
            llm_handler.model = None
            llm_handler.tokenizer = None
            llm_handler.generator = None
    
    try:
        # Generate response from LLM
        response: str = llm_handler.generate_response(
            prompt=request.prompt,
            max_length=request.max_length or 512,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.95
        )
        
        return ChatResponse(
            response=response,
            model=llm_handler.model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@router.post("/api/set-model", response_model=ModelChangeResponse)
async def set_model(request: ModelChangeRequest):
    """Change the active LLM model"""
    if request.model not in Config.AVAILABLE_MODELS:
        return ModelChangeResponse(
            success=False,
            message=f"Model '{request.model}' not found. Available models: {', '.join(Config.AVAILABLE_MODELS.keys())}"
        )
    
    try:
        model_name = Config.AVAILABLE_MODELS[request.model]["name"]
        
        # Only change model if different from current
        if model_name == llm_handler.model_name:
            return ModelChangeResponse(
                success=True,
                message=f"Already using model {request.model}"
            )
        
        # Update model and reinitialize
        os.environ["LLM_MODEL"] = request.model
        # We need to reset the model handler to load the new model
        llm_handler.model_name = model_name
        llm_handler.model = None
        llm_handler.tokenizer = None
        llm_handler.generator = None
        
        return ModelChangeResponse(
            success=True,
            message=f"Model changed to {request.model}"
        )
    except Exception as e:
        return ModelChangeResponse(
            success=False,
            message=f"Error changing model: {str(e)}"
        ) 