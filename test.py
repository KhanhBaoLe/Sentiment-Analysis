# train_deep_learning_force.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import json
import os

class DeepSentimentAnalyzer:
    def __init__(self, max_features=50000, max_len=200, embedding_dim=100):
        self.max_features = max_features
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_features, oov_token='<UNK>')
        self.model = None
        
    def build_model(self):
        self.model = Sequential([
            Embedding(self.max_features, self.embedding_dim, input_length=self.max_len),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.5),
            Bidirectional(LSTM(32)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def preprocess_data(self, texts, labels=None, training=False):
        if training:
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len)
        
        if labels is not None:
            labels = np.array([1 if label =='positive' else 0 for label in labels])
            return padded_sequences, labels
        
        return padded_sequences
    
    def train(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=64):
        print("Preprocessing training data...")
        x_train_processed, y_train_processed = self.preprocess_data(x_train, y_train, training=True)
        x_val_processed, y_val_processed = self.preprocess_data(x_val, y_val)
        
        print("Building model...")
        self.build_model()
        
        print("Training model...")
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint('dl_model.h5', save_best_only=True, monitor='val_accuracy')
        ]
        
        history = self.model.fit(
            x_train_processed, y_train_processed,
            validation_data=(x_val_processed, y_val_processed),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        
        # Save tokenizer
        with open('dl_tokenizer.json', 'w') as f:
            tokenizer_data = self.tokenizer.to_json()
            json.dump(tokenizer_data, f)
            
        # Save the final model
        self.model.save('dl_model.h5')
        print("✅ Model saved successfully!")
            
        return history

# Training script - FORCE TRAINING
if __name__ == "__main__":
    # Remove existing model files to force training
    if os.path.exists('dl_model.h5'):
        os.remove('dl_model.h5')
        print("Removed existing dl_model.h5")
    if os.path.exists('dl_tokenizer.json'):
        os.remove('dl_tokenizer.json')
        print("Removed existing dl_tokenizer.json")
    
    # Load data
    try:
        print("Loading data...")
        data = np.load('data_tokens.npz', allow_pickle=True)
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
        
        print(f"Training data: {len(x_train)} samples")
        print(f"Test data: {len(x_test)} samples")
        
        # Convert to raw text
        x_train_text = [' '.join(doc) if isinstance(doc, list) else doc for doc in x_train]
        x_test_text = [' '.join(doc) if isinstance(doc, list) else doc for doc in x_test]
        
        # Split validation
        split_idx = int(0.8 * len(x_train_text))
        x_train_split = x_train_text[:split_idx]
        y_train_split = y_train[:split_idx]
        x_val = x_train_text[split_idx:]
        y_val = y_train[split_idx:]
        
        print(f"Train split: {len(x_train_split)} samples")
        print(f"Val split: {len(x_val)} samples")
        
        # Train model with smaller parameters for testing
        dl_model = DeepSentimentAnalyzer(max_features=10000, max_len=100, embedding_dim=50)
        history = dl_model.train(x_train_split, y_train_split, x_val, y_val, epochs=5, batch_size=32)
        
        # Test the model immediately
        print("\nTesting the trained model...")
        test_texts = ["this film is good", "this movie is terrible"]
        for text in test_texts:
            sequences = dl_model.tokenizer.texts_to_sequences([text])
            padded_sequences = pad_sequences(sequences, maxlen=100)
            prediction = dl_model.model.predict(padded_sequences, verbose=0)[0][0]
            sentiment = "Positive" if prediction > 0.5 else "Negative"
            print(f"'{text}' -> {sentiment} ({prediction:.4f})")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()