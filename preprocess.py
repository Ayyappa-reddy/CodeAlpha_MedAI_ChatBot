#!/usr/bin/env python3
"""
Medical Chatbot Dataset Preprocessing Script
Handles ~228k doctor-patient conversations for semantic search
"""

import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
from typing import List, Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalDatasetPreprocessor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the preprocessor with sentence transformer model"""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        self.faiss_index = None
        self.patient_questions = []
        self.doctor_responses = []
        self.descriptions = []
        
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load the medical conversation dataset"""
        logger.info(f"Loading dataset from {file_path}")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
            
        logger.info(f"Loaded {len(df)} entries")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text using SpaCy (tokenization, lemmatization)"""
        if pd.isna(text) or text == '':
            return ""
        
        doc = self.nlp(str(text).lower().strip())
        
        # Tokenization and lemmatization (no stop word removal as dataset is cleaned)
        tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        
        return ' '.join(tokens)
    
    def extract_topics(self, texts: List[str], n_topics: int = 4) -> Dict:
        """Extract topics using LDA for FAQ categorization"""
        logger.info("Extracting topics for FAQ categorization...")
        
        # Vectorize the texts
        vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        # Apply LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract top words for each topic
        topics = {}
        topic_names = ["General Health", "Symptoms and Conditions", "Treatments", "Preventive Care"]
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics[topic_names[topic_idx]] = top_words
            
        return topics
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for all texts"""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for fast similarity search"""
        logger.info("Building FAISS index...")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index built with {index.ntotal} vectors")
        return index
    
    def process_dataset(self, file_path: str, output_dir: str = "processed_data"):
        """Main processing pipeline"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        df = self.load_dataset(file_path)
        
        # Preprocess texts
        logger.info("Preprocessing texts...")
        df['processed_patient'] = df['Patient'].apply(self.preprocess_text)
        df['processed_description'] = df['Description'].apply(self.preprocess_text)
        
        # Store original data
        self.patient_questions = df['Patient'].tolist()
        self.doctor_responses = df['Doctor'].tolist()
        self.descriptions = df['Description'].tolist()
        
        # Extract topics for categorization
        topics = self.extract_topics(df['processed_patient'].tolist())
        
        # Create embeddings for patient questions
        patient_embeddings = self.create_embeddings(df['processed_patient'].tolist())
        
        # Build FAISS index
        self.faiss_index = self.build_faiss_index(patient_embeddings)
        
        # Save processed data
        logger.info("Saving processed data...")
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, os.path.join(output_dir, "medical_faiss_index.bin"))
        
        # Save embeddings
        np.save(os.path.join(output_dir, "patient_embeddings.npy"), patient_embeddings)
        
        # Save metadata
        metadata = {
            'patient_questions': self.patient_questions,
            'doctor_responses': self.doctor_responses,
            'descriptions': self.descriptions,
            'topics': topics,
            'model_name': self.model_name,
            'total_entries': len(df)
        }
        
        with open(os.path.join(output_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save as JSON for easy access
        with open(os.path.join(output_dir, "medical_data.json"), 'w') as f:
            json.dump({
                'patient_questions': self.patient_questions,
                'doctor_responses': self.doctor_responses,
                'descriptions': self.descriptions,
                'topics': topics,
                'total_entries': len(df)
            }, f, indent=2)
        
        logger.info(f"Processing complete! Data saved to {output_dir}")
        logger.info(f"Total entries processed: {len(df)}")
        logger.info(f"Topics extracted: {list(topics.keys())}")
        
        return metadata

def main():
    """Main function to run preprocessing"""
    preprocessor = MedicalDatasetPreprocessor()
    
    # Check if dataset file exists
    dataset_files = ['medical_conversations.csv', 'medical_conversations.json', 'dataset.csv', 'dataset.json']
    dataset_path = None
    
    for file in dataset_files:
        if os.path.exists(file):
            dataset_path = file
            break
    
    if dataset_path is None:
        logger.error("No dataset file found. Please provide a CSV or JSON file with columns: Description, Patient, Doctor")
        logger.info("Creating sample dataset for testing...")
        
        # Create sample dataset for testing
        sample_data = {
            'Description': [
                'What does abutment of the nerve root mean?',
                'How to manage diabetes?',
                'Symptoms of hypertension',
                'Treatment for common cold',
                'Preventive care for heart disease'
            ],
            'Patient': [
                'Hi doctor, I am just wondering what is abutting and abutment of the nerve root means in a back issue...',
                'I was recently diagnosed with diabetes. What should I do to manage it?',
                'What are the common symptoms of high blood pressure?',
                'I have a cold with runny nose and cough. What can I do?',
                'How can I prevent heart disease?'
            ],
            'Doctor': [
                'Hi. I have gone through your query with diligence and would like you to know that I am here to help. Abutment means the nerve root is being pressed upon or compressed by surrounding structures...',
                'Managing diabetes involves a combination of diet, exercise, and medication. Here are some key steps...',
                'Common symptoms of hypertension include headache, shortness of breath, nosebleeds, and chest pain...',
                'For a common cold, rest, hydration, and over-the-counter medications can help relieve symptoms...',
                'Preventing heart disease involves maintaining a healthy diet, regular exercise, avoiding smoking...'
            ]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv('sample_medical_dataset.csv', index=False)
        dataset_path = 'sample_medical_dataset.csv'
        logger.info(f"Created sample dataset: {dataset_path}")
    
    # Process the dataset
    metadata = preprocessor.process_dataset(dataset_path)
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE!")
    print("="*50)
    print(f"Total entries processed: {metadata['total_entries']}")
    print(f"Model used: {metadata['model_name']}")
    print("Topics extracted:")
    for topic, words in metadata['topics'].items():
        print(f"  - {topic}: {', '.join(words[:5])}...")
    print("\nFiles created in 'processed_data' directory:")
    print("  - medical_faiss_index.bin (FAISS index)")
    print("  - patient_embeddings.npy (embeddings)")
    print("  - metadata.pkl (pickled metadata)")
    print("  - medical_data.json (JSON data)")
    print("\nReady to run the chatbot!")

if __name__ == "__main__":
    main()
