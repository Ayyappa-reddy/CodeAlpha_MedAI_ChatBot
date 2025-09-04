#!/usr/bin/env python3
"""
Medical Chatbot FastAPI Backend
Production-ready API with advanced features
"""

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
import json
import logging
from datetime import datetime
import os

from medical_chatbot import MedicalChatbot

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical AI Chatbot",
    description="Production-ready medical chatbot with advanced NLP and semantic search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
try:
    chatbot = MedicalChatbot()
    logger.info("Medical chatbot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {e}")
    chatbot = None

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    language: str = "en"

class SymptomCheckerRequest(BaseModel):
    symptom: str
    session_id: Optional[str] = None

class QuizRequest(BaseModel):
    question_index: int = 0
    answer: Optional[int] = None
    session_id: Optional[str] = None

class FAQRequest(BaseModel):
    question: str
    answer: str
    category: str
    password: str

# Session storage (in production, use Redis or database)
sessions = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/homepage", response_class=HTMLResponse)
async def homepage(request: Request):
    """Serve the mind-bending homepage"""
    return templates.TemplateResponse("homepage.html", {"request": request})

@app.get("/about-model", response_class=HTMLResponse)
async def about_model(request: Request):
    """Serve the About Model page"""
    return templates.TemplateResponse("about-model.html", {"request": request})

@app.get("/emergency", response_class=HTMLResponse)
async def emergency(request: Request):
    """Serve the Emergency Resources page"""
    return templates.TemplateResponse("emergency.html", {"request": request})

@app.get("/health-library", response_class=HTMLResponse)
async def health_library(request: Request):
    """Serve the Health Library page"""
    return templates.TemplateResponse("health-library.html", {"request": request})

@app.get("/resources", response_class=HTMLResponse)
async def resources(request: Request):
    """Serve the Resources page"""
    return templates.TemplateResponse("resources.html", {"request": request})

@app.get("/privacy-policy", response_class=HTMLResponse)
async def privacy_policy(request: Request):
    """Serve the Privacy Policy page"""
    return templates.TemplateResponse("privacy-policy.html", {"request": request})

@app.get("/terms-of-service", response_class=HTMLResponse)
async def terms_of_service(request: Request):
    """Serve the Terms of Service page"""
    return templates.TemplateResponse("terms-of-service.html", {"request": request})

@app.get("/medical-disclaimer", response_class=HTMLResponse)
async def medical_disclaimer(request: Request):
    """Serve the Medical Disclaimer page"""
    return templates.TemplateResponse("medical-disclaimer.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
async def contact(request: Request):
    """Serve the Contact page"""
    return templates.TemplateResponse("contact.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    """Serve the admin panel"""
    return templates.TemplateResponse("admin.html", {"request": request})

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with semantic matching"""
    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get response from chatbot
        response = chatbot.get_response(
            query=request.message,
            session_id=session_id,
            language=request.language
        )
        
        # Add session ID to response
        response['session_id'] = session_id
        
        # Log the interaction
        log_interaction(session_id, request.message, response)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/symptom-checker/start")
async def start_symptom_checker(request: SymptomCheckerRequest):
    """Start symptom checker flow"""
    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    try:
        session_id = request.session_id or str(uuid.uuid4())
        response = chatbot.start_symptom_checker(request.symptom)
        response['session_id'] = session_id
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error in symptom checker: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/symptom-checker/continue")
async def continue_symptom_checker(request: SymptomCheckerRequest):
    """Continue symptom checker flow"""
    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    try:
        session_id = request.session_id or str(uuid.uuid4())
        response = chatbot.continue_symptom_checker(request.symptom, request.symptom)
        response['session_id'] = session_id
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error in symptom checker: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/quiz/question/{question_index}")
async def get_quiz_question(question_index: int):
    """Get quiz question"""
    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    try:
        response = chatbot.get_quiz_question(question_index)
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error getting quiz question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/quiz/answer")
async def submit_quiz_answer(request: QuizRequest):
    """Submit quiz answer"""
    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    try:
        if request.answer is None:
            raise HTTPException(status_code=400, detail="Answer is required")
        
        response = chatbot.submit_quiz_answer(request.question_index, request.answer)
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error submitting quiz answer: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/topics")
async def get_topics():
    """Get available FAQ topics"""
    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    try:
        return JSONResponse(content={
            "topics": list(chatbot.topics.keys()),
            "topic_details": chatbot.topics
        })
        
    except Exception as e:
        logger.error(f"Error getting topics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/analytics")
async def get_analytics():
    """Get chatbot analytics"""
    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    try:
        analytics = chatbot.get_analytics()
        analytics['total_sessions'] = len(sessions)
        analytics['timestamp'] = datetime.now().isoformat()
        
        return JSONResponse(content=analytics)
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/faq/add")
async def add_faq(request: FAQRequest):
    """Add new FAQ (password protected)"""
    # Simple password protection (in production, use proper authentication)
    if request.password != "admin123":
        raise HTTPException(status_code=401, detail="Invalid password")
    
    try:
        # In production, save to database
        faq_data = {
            "question": request.question,
            "answer": request.answer,
            "category": request.category,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to JSON file (in production, use database)
        faqs_file = "faqs.json"
        faqs = []
        
        if os.path.exists(faqs_file):
            with open(faqs_file, 'r') as f:
                faqs = json.load(f)
        
        faqs.append(faq_data)
        
        with open(faqs_file, 'w') as f:
            json.dump(faqs, f, indent=2)
        
        return JSONResponse(content={"message": "FAQ added successfully", "faq": faq_data})
        
    except Exception as e:
        logger.error(f"Error adding FAQ: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/faq/list")
async def list_faqs():
    """List all custom FAQs"""
    try:
        faqs_file = "faqs.json"
        if os.path.exists(faqs_file):
            with open(faqs_file, 'r') as f:
                faqs = json.load(f)
            return JSONResponse(content={"faqs": faqs})
        else:
            return JSONResponse(content={"faqs": []})
            
    except Exception as e:
        logger.error(f"Error listing FAQs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy",
        "chatbot_initialized": chatbot is not None,
        "timestamp": datetime.now().isoformat()
    })

def log_interaction(session_id: str, message: str, response: Dict):
    """Log user interactions for analytics"""
    try:
        log_entry = {
            "session_id": session_id,
            "message": message,
            "response": response.get('response', ''),
            "confidence": response.get('confidence', 0.0),
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in session
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(log_entry)
        
        # Keep only last 10 interactions per session
        if len(sessions[session_id]) > 10:
            sessions[session_id] = sessions[session_id][-10:]
            
    except Exception as e:
        logger.error(f"Error logging interaction: {e}")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header"""
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
