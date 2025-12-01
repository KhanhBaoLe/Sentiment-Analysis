from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
import json
import re
import string
import requests
from bs4 import BeautifulSoup
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# Tạo thư mục uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ALLOWED_EXTENSIONS = {'txt', 'csv', 'json'}
ALLOWED_EXTENSIONS = {'txt', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============ LOAD MODELS ============
print("Loading models...")

try:
    # Load Scratch Model
    model_data = np.load('sentiment_model.npz', allow_pickle=True)
    embedding = model_data['embedding']
    W = model_data['W']
    b = model_data['b']

    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
        
    # Load Deep Learning Model
    dl_model, dl_tokenizer, MAX_LEN= None, None, 200
    
    if os.path.exists('dl_model.h5') and os.path.exists('dl_tokenizer.json'):
        try:
            dl_model = tf.keras.models.load_model('dl_model.h5')
            with open('dl_tokenizer.json', 'r') as f:
                tokenizer_data = json.load(f)
                dl_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)
            
            MAX_LEN = 100    
            print("✅ All models loaded successfully!")
        except Exception as e:
            print(f"⚠️  Error loading Deep Learning model: {e}")
            print("⚠️  Using Scratch model only")
            dl_model, dl_tokenizer = None, None
            
    else:
        print("⚠️  Deep Learning model not found, using Scratch model only")
        dl_model, dl_tokenizer = None, None
        
except Exception as e:
    print(f"❌ Error loading models: {e}")
    # Khởi tạo biến để tránh lỗi
    embedding, W, b, vocab, dl_model, dl_tokenizer, MAX_LEN = None, None, None, None, None, None, 200

# ============ UTILITY FUNCTIONS ============

def clean_text_simple(text):
    """Simple text cleaning"""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

def tokenize(text):
    """Simple tokenization"""
    return text.split()

def sentence_embedding_avg(tokens):
    """Compute average embedding"""
    if embedding is None:
        raise Exception("Scratch model not loaded")
    
    vecs = []
    unk_idx = vocab.get('<UNK>')
    for w in tokens:
        idx = vocab.get(w, unk_idx)       
        if idx is not None:
            vecs.append(embedding[idx])
    
    if not vecs:
        return np.zeros(embedding.shape[1])
    return np.mean(vecs, axis=0)

def softmax(z):
    """Softmax"""
    e = np.exp(z - np.max(z))
    return e / np.sum(e)

# ============ PREDICTION FUNCTIONS ============

def predict_scratch(review_text):
    """Make prediction on review"""
    if embedding is None:
        return {
            'prediction': 'Error',
            'confidence': 0.0,
            'probability_negative': 0.5,
            'probability_positive': 0.5,
            'model': 'Scratch',
            'error': 'Scratch model not loaded'
        }
    
    cleaned = clean_text_simple(review_text)
    tokens = tokenize(cleaned)
    
    emb = sentence_embedding_avg(tokens)
    logits = emb @ W + b
    probs = softmax(logits)
    
    prediction = int(np.argmax(probs))
    confidence = float(np.max(probs))
    
    return {
        'prediction': 'Positive' if prediction == 1 else 'Negative',
        'confidence': confidence,
        'probability_negative': float(probs[0]),
        'probability_positive': float(probs[1]),
        'model': 'Scratch'
    }

def predict_deep_learning(review_text):
    """Make prediction with Deep Learning model"""
    if dl_model is None:
        return {
            'prediction': 'Model Not Available',
            'confidence': 0.0,
            'probability_negative': 0.5,
            'probability_positive': 0.5,
            'model': 'Deep Learning',
            'error': 'Deep Learning model not loaded. Please train it first'
        }
        
    cleaned = clean_text_simple(review_text)
    
    #Preprocess for DL model
    sequence = dl_tokenizer.texts_to_sequences([cleaned])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)
    
    prediction = dl_model.predict(padded_sequence, verbose=0)[0][0]
    
    # prediction = 1 - prediction
    
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = prediction if sentiment == "Positive" else 1 - prediction
    
    return {
        'prediction': sentiment,
        'confidence': float(confidence),
        'probability_negative': float(1 - prediction),
        'probability_positive': float(prediction),
        'model': 'Deep Learning'
    }
    
def extract_text_from_url(url):
    """Extract text content from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:5000] # Limit to 5000 characters
    
    except Exception as e:
        raise Exception(f"Error extracting content from URL: {str(e)}")

# ============ Flask Routes ============

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        review = data.get('review', '')
        model_type = data.get('model', 'both')
        
        if not review.strip():
            return jsonify({'error': 'Empty review'}), 400
        
        results = {}
        
        if model_type in ['scratch', 'both']:
            results['scratch'] = predict_scratch(review)
            
        if model_type in ['dl', 'both']:
            results['deep_learning'] = predict_deep_learning(review)
            
        return jsonify(results)       
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/compare', methods=['POST'])
def compare_models():
    """Compare both models on the same text"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        review = data.get('review', '')
        
        if not review.strip():
            return jsonify({'error': 'Empty review'}), 400
        
        scratch_result = predict_scratch(review)
        dl_result = predict_deep_learning(review)
        
        # Kiểm tra nếu có lỗi trong kết quả
        agreement = False
        if 'error' not in scratch_result and 'error' not in dl_result:
            agreement = scratch_result['prediction'] == dl_result['prediction']
        elif 'error' in dl_result:
            # Nếu DL model không có, coi như chỉ có Scratch model 
            agreement = True 
            
        comparison = {
            'text': review,
            'scratch': scratch_result,
            'deep_learning': dl_result,
            'agreement': agreement
        }
        
        return jsonify(comparison)
    
    except Exception as e:
        return jsonify({'error': f'Comparison failed: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analyze"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(filename=file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read file based on extension
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
                # Assume first column contains text
                texts = df.iloc[:, 0].dropna().tolist()
            else: # txt file
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f if line.strip()]
                    
            # Analyze each text
            results = []
            for text in texts[:100]: # Limit to first 100 lines
                scratch_result = predict_scratch(text)
                dl_result = predict_deep_learning(text)
                
                agreement = False
                if 'error' not in scratch_result and 'error' not in dl_result:
                    agreement = scratch_result['prediction'] == dl_result['prediction']
                elif 'error' in dl_result:
                    agreement = True
                    
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'scratch': scratch_result,
                    'deep_learning': dl_result,
                    'agreement': agreement
                })
                
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'filename': filename,
                'total_reviews': len(results),
                'results': results
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    """Analyze content from URL"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        url = data.get('url', '')
        
        if not url.strip():
            return jsonify({'error': 'Empty URL'}), 400
    

        # Extract text from URL
        text_content = extract_text_from_url(url)
        
        if not text_content:
            return jsonify({'error': 'No content extracted from URL'}), 400
        
        # Analyze with both models
        scratch_result = predict_scratch(text_content)
        dl_result = predict_deep_learning(text_content)
        
        agreement = False
        if 'error' not in scratch_result and 'error' not in dl_result:
            agreement = scratch_result['prediction'] == dl_result['prediction']
        elif 'error' in dl_result:
            agreement = True
            
            
        return jsonify({
            'url': url,
            'content_preview': text_content[:200]+ '...',
            'scratch': scratch_result,
            'deep_learning': dl_result,
            'agreement': agreement
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple texts at once"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
        
        results = []
        for text in texts[:50]:
            if text and text.strip():
                scratch_result = predict_scratch(text)
                dl_result = predict_deep_learning(text)
                
                agreement = False
                if 'error' not in scratch_result and 'error' not in dl_result:
                    agreement = scratch_result['prediction'] == dl_result['prediction']
                elif 'error' in dl_result:
                    agreement = True
                
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'scratch': scratch_result,
                    'deep_learning': dl_result,
                    'agreement': agreement
                })
                
        return jsonify({
            'total_analyzed': len(results),
            'results': results
        })

    except Exception as e:
        return jsonify({'error': f'Batch analysis failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    scratch_loaded = embedding is not None
    dl_loaded = dl_model is not None
    
    return jsonify({
        'status': 'healthy' if (scratch_loaded or dl_loaded) else 'partial',
        'scratch_model_loaded': scratch_loaded,
        'deep_learning_model_loaded': dl_loaded,
        'vocab_size': len(vocab) if vocab else 0,
        'embedding_dim': embedding.shape[1] if embedding else 0
    })
if __name__ == '__main__':
    app.run(debug=True, port=5000)