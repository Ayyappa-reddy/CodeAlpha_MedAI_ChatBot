# 🏥 Medical AI Chatbot Model Training - Google Colab
# Copy each part to a separate cell in Colab

# ============================================================================
# 📦 PART 1: Setup and Installation
# ============================================================================
# Install required packages and download SpaCy model

"""
# Part 1: Install Required Packages
print("🚀 Installing required packages...")

# Install packages
!pip install sentence-transformers faiss-cpu spacy scikit-learn pandas numpy tqdm

# Download SpaCy English model
!python -m spacy download en_core_web_sm

print("✅ All packages installed successfully!")
print("🔄 Restart runtime to ensure all packages are loaded properly")
"""

# ============================================================================
# 📊 PART 2: Data Loading and Exploration
# ============================================================================
# Load your medical conversation dataset and explore its structure

"""
# Part 2: Data Loading and Exploration
import pandas as pd
import numpy as np
import json
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("📊 Loading and exploring dataset...")
print("=" * 50)

# 1. Load your dataset
def load_dataset(file_path):
    """Load medical conversation dataset"""
    print(f"📊 Loading dataset from: {file_path}")
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported format. Use CSV or JSON")
    
    print(f"✅ Loaded {len(df)} entries")
    print(f"📋 Columns: {df.columns.tolist()}")
    
    # Verify required columns
    required_cols = ['Description', 'Patient', 'Doctor']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove any empty rows
    df = df.dropna(subset=['Patient', 'Doctor'])
    print(f"🧹 Cleaned dataset: {len(df)} entries")
    
    return df

# Load your dataset here
# Replace 'your_dataset.csv' with your actual file path
dataset_path = 'your_dataset.csv'  # CHANGE THIS TO YOUR FILE PATH
df = load_dataset(dataset_path)

# Display sample data
print("\n📖 Sample Data:")
print(df.head(3))

# Basic statistics
print(f"\n📈 Dataset Statistics:")
print(f"Total conversations: {len(df)}")
print(f"Unique descriptions: {df['Description'].nunique()}")
print(f"Average patient question length: {df['Patient'].str.len().mean():.1f} characters")
print(f"Average doctor response length: {df['Doctor'].str.len().mean():.1f} characters")

# Check for any data quality issues
print(f"\n🔍 Data Quality Check:")
print(f"Empty patient questions: {df['Patient'].isna().sum()}")
print(f"Empty doctor responses: {df['Doctor'].isna().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
"""

# ============================================================================
# 🔧 PART 3: Text Preprocessing with SpaCy
# ============================================================================
# Clean and preprocess the text data for better embeddings

"""
# Part 3: Text Preprocessing with SpaCy
import spacy
from tqdm import tqdm

print("🔧 Setting up text preprocessing...")
print("=" * 50)

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ SpaCy model loaded successfully")
except OSError:
    print("❌ SpaCy model not found. Please run Part 1 first!")
    raise

def preprocess_text(text):
    """Preprocess text using SpaCy"""
    if pd.isna(text) or text == '':
        return ""
    
    # Convert to string and clean
    text = str(text).lower().strip()
    
    # Process with SpaCy
    doc = nlp(text)
    
    # Extract tokens (no stop word removal as dataset is cleaned)
    tokens = []
    for token in doc:
        if not token.is_punct and not token.is_space and not token.is_stop:
            tokens.append(token.lemma_)
    
    return ' '.join(tokens)

def batch_preprocess(texts, batch_size=1000):
    """Process texts in batches for efficiency"""
    processed_texts = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Preprocessing texts"):
        batch = texts[i:i + batch_size]
        processed_batch = [preprocess_text(text) for text in batch]
        processed_texts.extend(processed_batch)
    
    return processed_texts

print("\n🔄 Preprocessing patient questions...")
df['processed_patient'] = batch_preprocess(df['Patient'].tolist())

print("\n🔄 Preprocessing descriptions...")
df['processed_description'] = batch_preprocess(df['Description'].tolist())

# Save processed data
df.to_csv('processed_dataset.csv', index=False)
print("💾 Saved processed dataset to 'processed_dataset.csv'")

# Display preprocessing results
print(f"\n📊 Preprocessing Results:")
print(f"Sample processed patient question: {df['processed_patient'].iloc[0][:100]}...")
print(f"Sample processed description: {df['processed_description'].iloc[0][:100]}...")

# Check preprocessing quality
print(f"\n🔍 Preprocessing Quality Check:")
print(f"Empty processed patient questions: {df['processed_patient'].isna().sum()}")
print(f"Empty processed descriptions: {df['processed_description'].isna().sum()}")
print(f"Average processed length: {df['processed_patient'].str.len().mean():.1f} tokens")
"""

# ============================================================================
# 🧠 PART 4: Generate Embeddings with Sentence Transformers
# ============================================================================
# Create semantic embeddings for all texts using the transformer model

"""
# Part 4: Generate Embeddings with Sentence Transformers
from sentence_transformers import SentenceTransformer
import torch

print("🧠 Setting up sentence transformer model...")
print("=" * 50)

# Initialize the model
model_name = 'all-MiniLM-L6-v2'  # Lightweight but effective model
model = SentenceTransformer(model_name)

print(f"✅ Model loaded: {model_name}")
print(f"📏 Embedding dimension: {model.get_sentence_embedding_dimension()}")
print(f"💾 Model size: {sum(p.numel() for p in model.parameters()):,} parameters")

def generate_embeddings(texts, batch_size=32):
    """Generate embeddings for texts"""
    print(f"🔄 Generating embeddings for {len(texts)} texts...")
    
    embeddings = model.encode(
        texts, 
        show_progress_bar=True, 
        batch_size=batch_size,
        convert_to_numpy=True
    )
    
    print(f"✅ Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    return embeddings

# Generate embeddings for patient questions
print("\n🔄 Generating patient question embeddings...")
patient_embeddings = generate_embeddings(df['processed_patient'].tolist())

# Generate embeddings for descriptions
print("\n🔄 Generating description embeddings...")
description_embeddings = generate_embeddings(df['processed_description'].tolist())

# Save embeddings
np.save('patient_embeddings.npy', patient_embeddings)
np.save('description_embeddings.npy', description_embeddings)

print("💾 Saved embeddings to .npy files")

# Display embedding statistics
print(f"\n📊 Embedding Statistics:")
print(f"Patient embeddings shape: {patient_embeddings.shape}")
print(f"Description embeddings shape: {description_embeddings.shape}")
print(f"Embedding sample (first 5 values): {patient_embeddings[0][:5]}")
print(f"Memory usage: {patient_embeddings.nbytes / 1024 / 1024:.1f} MB")

# Test embedding quality
print(f"\n🧪 Testing embedding quality...")
test_texts = [
    "What are the symptoms of diabetes?",
    "How to manage diabetes?",
    "What is a healthy diet?"
]
test_embeddings = model.encode(test_texts)
print(f"✅ Test embeddings generated successfully")
"""

# ============================================================================
# 🔍 PART 5: Build FAISS Index for Fast Similarity Search
# ============================================================================
# Create FAISS indices for efficient similarity search

"""
# Part 5: Build FAISS Index for Fast Similarity Search
import faiss

print("🔍 Building FAISS index for fast similarity search...")
print("=" * 50)

def build_faiss_index(embeddings, index_type='cosine'):
    """Build FAISS index for embeddings"""
    print(f"🏗️ Building {index_type} similarity index...")
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    
    if index_type == 'cosine':
        # Inner product for cosine similarity (after normalization)
        index = faiss.IndexFlatIP(dimension)
    else:
        # L2 distance
        index = faiss.IndexFlatL2(dimension)
    
    # Add vectors to index
    index.add(embeddings.astype('float32'))
    
    print(f"✅ FAISS index built successfully")
    print(f"📏 Index dimension: {dimension}")
    print(f"🔢 Total vectors: {index.ntotal}")
    
    return index

# Build index for patient questions
patient_index = build_faiss_index(patient_embeddings, 'cosine')

# Build index for descriptions
description_index = build_faiss_index(description_embeddings, 'cosine')

# Save FAISS indices
faiss.write_index(patient_index, 'patient_faiss_index.bin')
faiss.write_index(description_index, 'description_faiss_index.bin')

print("💾 Saved FAISS indices to .bin files")

# Test the index
print("\n🧪 Testing FAISS index...")
test_query = "What are the symptoms of diabetes?"
test_embedding = model.encode([test_query])
faiss.normalize_L2(test_embedding)

# Search for similar questions
D, I = patient_index.search(test_embedding.astype('float32'), 5)

print(f"🔍 Test query: '{test_query}'")
print(f"📊 Top 5 similar questions:")
for i, (distance, idx) in enumerate(zip(D[0], I[0])):
    similarity = (distance + 1) / 2  # Convert to 0-1 scale
    print(f"  {i+1}. Similarity: {similarity:.3f} - {df['Patient'].iloc[idx][:80]}...")

# Performance test
print(f"\n⚡ Performance Test:")
import time
start_time = time.time()
for _ in range(100):
    D, I = patient_index.search(test_embedding.astype('float32'), 5)
end_time = time.time()
avg_time = (end_time - start_time) / 100 * 1000  # Convert to milliseconds
print(f"Average search time: {avg_time:.2f}ms per query")
"""

# ============================================================================
# 🏷️ PART 6: Topic Modeling with LDA
# ============================================================================
# Extract topics from the medical conversations for better categorization

"""
# Part 6: Topic Modeling with LDA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

print("🏷️ Performing topic modeling...")
print("=" * 50)

def extract_topics(texts, n_topics=4):
    """Extract topics using LDA"""
    print(f"🔍 Extracting {n_topics} topics...")
    
    # Vectorize texts
    vectorizer = CountVectorizer(
        max_features=1000, 
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    X = vectorizer.fit_transform(texts)
    
    # Apply LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics, 
        random_state=42,
        max_iter=20,
        learning_method='batch'
    )
    
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
    
    return topics, vectorizer, lda

# Extract topics from patient questions
topics, vectorizer, lda_model = extract_topics(df['processed_patient'].tolist())

print("\n🏷️ Extracted Topics:")
for topic, words in topics.items():
    print(f"  {topic}: {', '.join(words[:5])}...")

# Save topic model components
with open('topic_model.pkl', 'wb') as f:
    pickle.dump({
        'vectorizer': vectorizer,
        'lda_model': lda_model,
        'topics': topics
    }, f)

print("💾 Saved topic model to 'topic_model.pkl'")

# Test topic assignment
print(f"\n🧪 Testing topic assignment...")
test_texts = [
    "diabetes symptoms and treatment",
    "heart disease prevention tips",
    "common cold remedies",
    "healthy diet recommendations"
]

for text in test_texts:
    # Preprocess
    processed = preprocess_text(text)
    # Vectorize
    vectorized = vectorizer.transform([processed])
    # Get topic
    topic_dist = lda_model.transform(vectorized)[0]
    topic_idx = topic_dist.argmax()
    topic_name = list(topics.keys())[topic_idx]
    confidence = topic_dist[topic_idx]
    
    print(f"  '{text}' → {topic_name} (confidence: {confidence:.3f})")
"""

# ============================================================================
# 💾 PART 7: Final Export and Model Summary
# ============================================================================
# Save all model files and create a summary for the chatbot

"""
# Part 7: Final Export and Model Summary
print("💾 Final export and model summary...")
print("=" * 50)

# Prepare final metadata
metadata = {
    'patient_questions': df['Patient'].tolist(),
    'doctor_responses': df['Doctor'].tolist(),
    'descriptions': df['Description'].tolist(),
    'topics': topics,
    'model_name': model_name,
    'total_entries': len(df),
    'embedding_dimension': patient_embeddings.shape[1],
    'created_at': datetime.now().isoformat(),
    'dataset_info': {
        'total_conversations': len(df),
        'unique_descriptions': df['Description'].nunique(),
        'avg_patient_length': df['Patient'].str.len().mean(),
        'avg_doctor_length': df['Doctor'].str.len().mean()
    }
}

# Save metadata
with open('medical_chatbot_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

# Save as JSON for easy access
with open('medical_chatbot_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n🎉 Model Training Complete!")
print("=" * 50)
print("📁 Generated Files:")
print("  - patient_embeddings.npy")
print("  - description_embeddings.npy")
print("  - patient_faiss_index.bin")
print("  - description_faiss_index.bin")
print("  - topic_model.pkl")
print("  - medical_chatbot_metadata.pkl")
print("  - medical_chatbot_metadata.json")
print("  - processed_dataset.csv")

print(f"\n📊 Final Statistics:")
print(f"  Total conversations: {len(df)}")
print(f"  Embedding dimension: {patient_embeddings.shape[1]}")
print(f"  Topics extracted: {len(topics)}")
print(f"  FAISS index size: {patient_index.ntotal}")
print(f"  Total model size: {(patient_embeddings.nbytes + description_embeddings.nbytes) / 1024 / 1024:.1f} MB")

# Create a simple test function
def test_semantic_search(query, top_k=5):
    """Test semantic search functionality"""
    # Preprocess query
    processed_query = preprocess_text(query)
    # Generate embedding
    query_embedding = model.encode([processed_query])
    faiss.normalize_L2(query_embedding)
    
    # Search
    D, I = patient_index.search(query_embedding.astype('float32'), top_k)
    
    results = []
    for i, (distance, idx) in enumerate(zip(D[0], I[0])):
        similarity = (distance + 1) / 2
        results.append({
            'rank': i + 1,
            'similarity': similarity,
            'question': df['Patient'].iloc[idx],
            'answer': df['Doctor'].iloc[idx][:200] + '...'
        })
    
    return results

print(f"\n🧪 Testing semantic search...")
test_queries = [
    "What are the symptoms of diabetes?",
    "How to lower blood pressure?",
    "What is a healthy diet?"
]

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    results = test_semantic_search(query, 3)
    for result in results:
        print(f"  {result['rank']}. Similarity: {result['similarity']:.3f}")
        print(f"     Q: {result['question'][:80]}...")
        print(f"     A: {result['answer']}")

print("\n🚀 Next steps:")
print("1. Download all generated files from Colab")
print("2. Copy files to your chatbot project directory")
print("3. Update file paths in medical_chatbot.py")
print("4. Run the chatbot with your trained model!")

print("\n✅ Training complete! Your model is ready for the chatbot.")
"""

# ============================================================================
# 📥 DOWNLOAD INSTRUCTIONS
# ============================================================================
"""
## 📥 Download Instructions

After running all parts successfully, download these files:

1. **patient_embeddings.npy** - Patient question embeddings
2. **description_embeddings.npy** - Description embeddings  
3. **patient_faiss_index.bin** - FAISS index for patient questions
4. **description_faiss_index.bin** - FAISS index for descriptions
5. **topic_model.pkl** - Topic modeling components
6. **medical_chatbot_metadata.pkl** - Main metadata file
7. **medical_chatbot_metadata.json** - JSON version of metadata
8. **processed_dataset.csv** - Preprocessed dataset

**Right-click each file → Download** or use the Colab file browser.

Then copy all files to your chatbot project directory!
"""

print("📚 Colab notebook code ready!")
print("Copy each part to separate cells in Google Colab")
print("Make sure to upload your dataset file first!")
