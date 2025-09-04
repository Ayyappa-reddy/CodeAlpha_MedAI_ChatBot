#!/usr/bin/env python3
"""
Medical Chatbot Core Engine
Handles semantic matching, symptom checking, and advanced features
"""

import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import spacy
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import os
import requests
# Azure Translator removed - no longer needed

logger = logging.getLogger(__name__)

class MedicalChatbot:
    def __init__(self, data_dir: str = "models"):
        """Initialize the medical chatbot with preprocessed data"""
        self.data_dir = data_dir
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load preprocessed data
        self.load_data()
        
        # Translation support removed - no longer needed
        
        # Confidence threshold
        self.confidence_threshold = 0.6
        
        # Session storage for context
        self.sessions = {}
        
        # Health resource links
        self.health_resources = {
            "diabetes": "https://www.mayoclinic.org/diseases-conditions/diabetes",
            "hypertension": "https://www.mayoclinic.org/diseases-conditions/high-blood-pressure",
            "heart": "https://www.mayoclinic.org/diseases-conditions/heart-disease",
            "cancer": "https://www.mayoclinic.org/diseases-conditions/cancer",
            "mental": "https://www.mayoclinic.org/diseases-conditions/mental-illness",
            "general": "https://www.cdc.gov/healthinfo/"
        }
        
        # Symptom checker decision tree
        self.symptom_checker = self._build_symptom_checker()
        
        # Educational quiz questions
        self.quiz_questions = self._build_quiz_questions()
        
    # Translation setup removed - no longer needed
    
    def load_data(self):
        """Load preprocessed data and FAISS index"""
        try:
            # Load FAISS index for patient questions
            self.faiss_index = faiss.read_index(os.path.join(self.data_dir, "patient_faiss_index.bin"))
            
            # Load metadata
            with open(os.path.join(self.data_dir, "medical_chatbot_metadata.pkl"), 'rb') as f:
                metadata = pickle.load(f)
            
            self.patient_questions = metadata['patient_questions']
            self.doctor_responses = metadata['doctor_responses']
            self.descriptions = metadata['descriptions']
            self.topics = metadata['topics']
            
            logger.info(f"Loaded {len(self.patient_questions)} medical conversations")
            logger.info(f"FAISS index contains {self.faiss_index.ntotal} vectors")
            
        except FileNotFoundError:
            logger.error("Preprocessed data not found. Please run preprocess.py first.")
            raise
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess user query using SpaCy"""
        if not query or query.strip() == "":
            return ""
        
        doc = self.nlp(query.lower().strip())
        tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        return ' '.join(tokens)
    
    # Translation methods removed - no longer needed
    
    def find_best_match(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Find best matching medical responses using FAISS"""
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        if not processed_query:
            return []
        
        # Generate embedding for query
        query_embedding = self.model.encode([processed_query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        similarities, indices = self.faiss_index.search(
            query_embedding.astype('float32'), top_k
        )
        
        # Convert similarities to confidence scores (0-1)
        confidence_scores = (similarities[0] + 1) / 2
        
        # Return results with confidence scores
        results = []
        for idx, confidence in zip(indices[0], confidence_scores):
            if idx < len(self.doctor_responses):
                # Convert numpy float32 to Python float for JSON serialization
                results.append((idx, float(confidence), self.doctor_responses[idx]))
        
        return results
    
    def get_response(self, query: str, session_id: str = None, language: str = 'en') -> Dict:
        """Get medical response with confidence scoring and context awareness"""
        # Language support simplified - only English supported
        original_query = query
        
        # Check for conversational/greeting queries first
        conversational_response = self._handle_conversational_query(query)
        if conversational_response:
            return conversational_response
        
        # Find best matches for medical queries
        matches = self.find_best_match(query)
        
        if not matches:
            return {
                'response': "I'm not sure about that. Could you please rephrase your question or consult a healthcare professional for medical advice.",
                'confidence': 0.0,
                'suggestions': self._get_suggestions(query),
                'disclaimer': "This is for informational purposes only. Please consult a healthcare professional for medical advice."
            }
        
        best_match_idx, confidence, response = matches[0]
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return {
                'response': "I'm not entirely sure about that. Could you try rephrasing your question? For specific medical advice, please consult a healthcare professional.",
                'confidence': confidence,
                'suggestions': self._get_suggestions(query),
                'disclaimer': "This is for informational purposes only. Please consult a healthcare professional for medical advice."
            }
        
        # Store context for session
        if session_id:
            self._update_session_context(session_id, query, best_match_idx)
        
        # Get related resources
        resources = self._get_health_resources(query)
        
        # Get related questions
        related_questions = self._get_related_questions(best_match_idx)
        
        # Language support simplified - only English supported
        
        return {
            'response': response,
            'confidence': confidence,
            'confidence_percentage': f"{confidence * 100:.1f}%",
            'related_questions': related_questions,
            'health_resources': resources,
            'suggestions': self._get_suggestions(query),
            'disclaimer': "This is for informational purposes only. Please consult a healthcare professional for medical advice."
        }
    
    def _update_session_context(self, session_id: str, query: str, response_idx: int):
        """Update session context for related question suggestions"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({
            'query': query,
            'response_idx': response_idx,
            'timestamp': datetime.now()
        })
        
        # Keep only last 3 interactions
        if len(self.sessions[session_id]) > 3:
            self.sessions[session_id] = self.sessions[session_id][-3:]
    
    def _get_suggestions(self, query: str) -> List[str]:
        """Get suggested questions based on query"""
        # Simple keyword-based suggestions
        suggestions = []
        
        if 'diabetes' in query.lower():
            suggestions.extend([
                "How to manage diabetes?",
                "What are the symptoms of diabetes?",
                "Diabetes diet recommendations"
            ])
        elif 'blood pressure' in query.lower() or 'hypertension' in query.lower():
            suggestions.extend([
                "How to lower blood pressure naturally?",
                "Blood pressure medication side effects",
                "Hypertension risk factors"
            ])
        elif 'heart' in query.lower():
            suggestions.extend([
                "Heart disease prevention",
                "Heart attack symptoms",
                "Cardiovascular health tips"
            ])
        else:
            suggestions.extend([
                "Common health symptoms",
                "Preventive care tips",
                "When to see a doctor"
            ])
        
        return suggestions[:3]
    
    def _get_related_questions(self, response_idx: int) -> List[str]:
        """Get related questions based on current response"""
        # Find questions with similar topics
        related = []
        
        # Simple approach: find questions with similar keywords
        current_desc = self.descriptions[response_idx].lower()
        
        for i, desc in enumerate(self.descriptions):
            if i != response_idx:
                # Simple keyword overlap
                current_words = set(current_desc.split())
                desc_words = set(desc.lower().split())
                overlap = len(current_words.intersection(desc_words))
                
                if overlap >= 2:  # At least 2 common words
                    related.append(self.patient_questions[i])
                
                if len(related) >= 3:
                    break
        
        return related
    
    def _get_health_resources(self, query: str) -> List[Dict]:
        """Get relevant health resource links"""
        resources = []
        query_lower = query.lower()
        
        for keyword, url in self.health_resources.items():
            if keyword in query_lower:
                resources.append({
                    'title': f"{keyword.title()} Information",
                    'url': url,
                    'source': 'Mayo Clinic' if 'mayoclinic' in url else 'CDC'
                })
        
        if not resources:
            resources.append({
                'title': 'General Health Information',
                'url': self.health_resources['general'],
                'source': 'CDC'
            })
        
        return resources
    
    def _build_symptom_checker(self) -> Dict:
        """Build symptom checker decision tree"""
        return {
            "chest_pain": {
                "question": "What type of chest pain are you experiencing?",
                "options": {
                    "sharp": {
                        "question": "Is the pain worse with breathing?",
                        "options": {
                            "yes": "This could be pleurisy or a lung condition. Please consult a doctor immediately.",
                            "no": "Sharp chest pain could indicate various conditions. Please seek medical attention."
                        }
                    },
                    "dull": {
                        "question": "Is the pain accompanied by shortness of breath?",
                        "options": {
                            "yes": "This could be a heart condition. Please seek emergency medical care.",
                            "no": "Dull chest pain should be evaluated by a doctor."
                        }
                    }
                }
            },
            "headache": {
                "question": "How severe is your headache?",
                "options": {
                    "mild": "Try rest, hydration, and over-the-counter pain relievers. If persistent, see a doctor.",
                    "severe": "Severe headaches could indicate serious conditions. Please consult a doctor immediately.",
                    "migraine": "Consider migraine-specific treatments and consult a doctor for proper management."
                }
            }
        }
    
    def _build_quiz_questions(self) -> List[Dict]:
        """Build educational quiz questions"""
        return [
            {
                "question": "What is a common symptom of hypertension?",
                "options": ["Headache", "Rash", "Joint pain", "Hair loss"],
                "correct": 0,
                "explanation": "Headache is a common symptom of high blood pressure."
            },
            {
                "question": "Which of the following is a risk factor for diabetes?",
                "options": ["Family history", "Eye color", "Hair length", "Shoe size"],
                "correct": 0,
                "explanation": "Family history is a significant risk factor for diabetes."
            },
            {
                "question": "What is the recommended daily water intake?",
                "options": ["2-3 cups", "4-6 cups", "8-10 cups", "12-15 cups"],
                "correct": 2,
                "explanation": "Most adults should drink 8-10 cups of water daily."
            }
        ]
    
    def start_symptom_checker(self, symptom: str) -> Dict:
        """Start symptom checker flow"""
        symptom_lower = symptom.lower()
        
        for key, data in self.symptom_checker.items():
            if key in symptom_lower or any(word in symptom_lower for word in key.split('_')):
                return {
                    'type': 'symptom_checker',
                    'current_symptom': key,
                    'question': data['question'],
                    'options': list(data['options'].keys())
                }
        
        return {
            'type': 'symptom_checker',
            'response': "I can help you with common symptoms like chest pain, headache, fever, etc. Please describe your symptom."
        }
    
    def continue_symptom_checker(self, symptom: str, answer: str) -> Dict:
        """Continue symptom checker flow"""
        # Simplified implementation - in production, this would be more sophisticated
        if 'chest' in symptom.lower() and 'sharp' in answer.lower():
            return {
                'type': 'symptom_checker',
                'response': "Sharp chest pain could indicate various conditions. Please consult a doctor immediately.",
                'recommendation': "Seek medical attention"
            }
        elif 'headache' in symptom.lower():
            return {
                'type': 'symptom_checker',
                'response': "Try rest, hydration, and over-the-counter pain relievers. If persistent, see a doctor.",
                'recommendation': "Self-care with medical follow-up if needed"
            }
        
        return {
            'type': 'symptom_checker',
            'response': "Based on your symptoms, I recommend consulting a healthcare professional for proper evaluation.",
            'recommendation': "Medical consultation"
        }
    
    def get_quiz_question(self, question_index: int = 0) -> Dict:
        """Get quiz question"""
        if question_index >= len(self.quiz_questions):
            return {'type': 'quiz_complete', 'message': 'Quiz completed!'}
        
        return {
            'type': 'quiz',
            'question_index': question_index,
            'question': self.quiz_questions[question_index]
        }
    
    def submit_quiz_answer(self, question_index: int, answer: int) -> Dict:
        """Submit quiz answer and get feedback"""
        if question_index >= len(self.quiz_questions):
            return {'error': 'Invalid question index'}
        
        question = self.quiz_questions[question_index]
        is_correct = answer == question['correct']
        
        return {
            'type': 'quiz_feedback',
            'correct': is_correct,
            'explanation': question['explanation'],
            'next_question': question_index + 1 if question_index + 1 < len(self.quiz_questions) else None
        }
    
    def _handle_conversational_query(self, query: str) -> Optional[Dict]:
        """Handle conversational queries like greetings, thanks, etc."""
        query_lower = query.lower().strip()
        
        # Greeting patterns
        greetings = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            'hi there', 'hello there', 'hey there', 'greetings', 'howdy'
        ]
        
        # Farewell patterns
        farewells = [
            'bye', 'goodbye', 'see you', 'farewell', 'take care', 'have a good day',
            'thanks bye', 'thank you bye', 'bye bye', 'see you later', 'talk to you later'
        ]
        
        # Thanks patterns
        thanks = [
            'thank you', 'thanks', 'thank you so much', 'thanks a lot', 'appreciate it',
            'much appreciated', 'grateful', 'thank you very much'
        ]
        
        # How are you patterns
        how_are_you = [
            'how are you', 'how are you doing', 'how do you do', 'what\'s up', 'how\'s it going'
        ]
        
        # Check for greetings
        if any(greeting in query_lower for greeting in greetings):
            responses = [
                "Hello! I'm MedAI, your medical assistant. How can I help you with your health questions today?",
                "Hi there! I'm here to help with your medical inquiries. What would you like to know?",
                "Hello! I'm trained on thousands of medical conversations to help answer your health questions. What can I assist you with?",
                "Hi! I'm your medical AI assistant. Feel free to ask me about symptoms, treatments, or any health concerns you might have."
            ]
            import random
            return {
                'response': random.choice(responses),
                'confidence': 1.0,
                'confidence_percentage': '100%',
                'type': 'greeting',
                'disclaimer': "This is for informational purposes only. Please consult a healthcare professional for medical advice."
            }
        
        # Check for farewells
        if any(farewell in query_lower for farewell in farewells):
            responses = [
                "Goodbye! Take care of your health and don't hesitate to reach out if you have more questions.",
                "Bye! Remember to consult healthcare professionals for any serious health concerns.",
                "Take care! I'm here whenever you need medical information or have health questions.",
                "Goodbye! Stay healthy and feel free to ask me anything about your health anytime."
            ]
            import random
            return {
                'response': random.choice(responses),
                'confidence': 1.0,
                'confidence_percentage': '100%',
                'type': 'farewell',
                'disclaimer': "This is for informational purposes only. Please consult a healthcare professional for medical advice."
            }
        
        # Check for thanks
        if any(thank in query_lower for thank in thanks):
            responses = [
                "You're welcome! I'm glad I could help. Feel free to ask if you have any other health questions.",
                "My pleasure! I'm here to assist with your medical inquiries anytime.",
                "You're very welcome! Don't hesitate to reach out if you need more health information.",
                "Happy to help! Remember to consult healthcare professionals for medical advice and treatment."
            ]
            import random
            return {
                'response': random.choice(responses),
                'confidence': 1.0,
                'confidence_percentage': '100%',
                'type': 'thanks',
                'disclaimer': "This is for informational purposes only. Please consult a healthcare professional for medical advice."
            }
        
        # Check for "how are you"
        if any(how in query_lower for how in how_are_you):
            responses = [
                "I'm doing well, thank you for asking! I'm here and ready to help with your medical questions. How can I assist you today?",
                "I'm functioning perfectly and ready to help! What health questions do you have for me?",
                "I'm doing great! I'm here to provide medical information and answer your health-related questions. What would you like to know?",
                "I'm well, thank you! I'm always ready to help with medical inquiries. What can I help you with today?"
            ]
            import random
            return {
                'response': random.choice(responses),
                'confidence': 1.0,
                'confidence_percentage': '100%',
                'type': 'how_are_you',
                'disclaimer': "This is for informational purposes only. Please consult a healthcare professional for medical advice."
            }
        
        # Check for very short queries that are likely not medical
        if len(query_lower.split()) <= 2 and not any(medical_word in query_lower for medical_word in [
            'pain', 'hurt', 'ache', 'sick', 'fever', 'cough', 'headache', 'stomach', 'chest', 'back',
            'symptom', 'disease', 'condition', 'medicine', 'drug', 'treatment', 'doctor', 'hospital'
        ]):
            return {
                'response': "I'm a medical AI assistant designed to help with health-related questions. Could you please ask me about symptoms, treatments, or any medical concerns you might have?",
                'confidence': 0.5,
                'confidence_percentage': '50%',
                'type': 'clarification',
                'disclaimer': "This is for informational purposes only. Please consult a healthcare professional for medical advice."
            }
        
        return None
    
    def get_analytics(self) -> Dict:
        """Get basic analytics"""
        return {
            'total_conversations': len(self.patient_questions),
            'topics': list(self.topics.keys()),
            'faiss_index_size': self.faiss_index.ntotal,
            'active_sessions': len(self.sessions)
        }
