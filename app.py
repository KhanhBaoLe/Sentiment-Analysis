from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import re
import string

app = Flask(__name__)

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

print("Loading model and vocabulary...")
model_data = np.load('sentiment_model.npz', allow_pickle=True)
embedding = model_data['embedding']
W = model_data['W']
b = model_data['b']

with open('vocab.json', 'r') as f:
    vocab = json.load(f)

def sentence_embedding_avg(tokens):
    """Compute average embedding"""
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

def predict(review_text):
    """Make prediction on review"""
    cleaned = clean_text_simple(review_text)
    tokens = tokenize(cleaned)
    
    emb = sentence_embedding_avg(tokens)
    logits = emb @ W + b
    probs = softmax(logits)
    
    prediction = int(np.argmax(probs))
    confidence = float(np.max(probs))
    
    return prediction, confidence, probs

# ============ Flask Routes ============

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.json
        review = data.get('review', '')
        
        if not review.strip():
            return jsonify({'error': 'Empty review'}), 400
        
        pred, conf, probs = predict(review)
        label = 'Positive' if pred == 1 else 'Negative'
        
        return jsonify({
            'prediction': label,
            'confidence': conf,
            'probability_negative': float(probs[0]),
            'probability_positive': float(probs[1])
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'vocab_size': len(vocab),
        'embedding_dim': embedding.shape[1]
    })
if __name__ == '__main__':
    app.run(debug=True, port=5000)