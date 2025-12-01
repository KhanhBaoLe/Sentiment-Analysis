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
        # Xây dựng model LSTM
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
        # Tiền xử lý dữ liệu
        if training:
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len)
        
        if labels is not None:
            labels = np.array([1 if label =='positive' else 0 for label in labels])
            return padded_sequences, labels
        
        return padded_sequences
    
    def train(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=64):
        # Huấn luyện model
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
        
        # Lưu tokenizer
        with open('dl_tokenizer.json', 'w') as f:
            tokenizer_data = self.tokenizer.to_json()
            json.dump(tokenizer_data, f)
        
        # Save the final model explicitly
        self.model.save('dl_model.h5')
        print("Model saved successfully!")
            
        return history
    
    def load_model(self):
        """Load model ad tokenizer"""
        try:
            self.model = tf.keras.models.load_model('dl_model.h5')
            with open('dl_tokenizer.json', 'r') as f:
                tokenizer_data = json.load(f)
                self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)                
            print("Deep Learning model loaded successfully!")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, texts):
        # Dự đoán sentiment
        if not self.model:
            if not self.load_model():
                raise Exception("Failed to load model")
            
        processed_texts = self.preprocess_data(texts)
        predictions = self.model.predict(processed_texts)
        
        results = []
        for pred in predictions:
            sentiment = 'Positive' if pred[0] > 0.5 else 'Negative'
            confidence = float(pred[0] if sentiment == "Positive" else float(1 - pred[0]))
            
            results.append({
                'sentiment': sentiment,
                'confidence': confidence,
                'probability': float(pred[0])
            })
        
        return results
    
    def evaluate(self, x_test, y_test):
        # Đánh giá model
        x_test_processed, y_test_processed = self.prepocess_data(x_test, y_test)
        loss, accuracy = self.model.evaluate(x_test_processed, y_test_processed)
        
        predictions = self.model.predict(x_test_processed)
        pred_labels = (predictions > 0.5).astype(int).flatten()
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test_processed, pred_labels)}")
        print(f"Classification Report:\n{classification_report(y_test_processed,pred_labels)}")
        
        return accuracy 
    
# Training script
if __name__ == "__main__":
    # Check if model already exists
    if os.path.join('dl_model.h5') and os.path.exists('dl_tokenizer.json'):
        print("Model already exists. Training skipped.")
    else:
        try:
            # Load data
            data = np.load('data_tokens.npz', allow_pickle=True)
            x_train, y_train = data['x_train'], data['y_train']
            x_test, y_test = data['x_test'], data['y_test']
            
            # Convert to raw test
            x_train_text = [' '.join(doc) if isinstance(doc, list) else doc for doc in x_train]
            x_test_text = [' '.join(doc) if isinstance(doc, list) else doc for doc in x_test]
            
            # Split validation
            split_idx = int(0.8 * len(x_train_text))
            x_train_split = x_train_text[:split_idx]
            y_train_split = y_train[:split_idx]
            x_val = x_train_text[split_idx:]
            y_val = y_train[split_idx:]
            
            # Train model
            dl_model = DeepSentimentAnalyzer(max_features=30000, max_len=200, embedding_dim=100)
            history = dl_model.train(x_train_split, y_train_split, x_val, y_val, epochs=15, batch_size=64)
            
            # Evaluate
            print("\nEvaluating on test set...")
            dl_model.evaluate(x_test_text, y_test)
            
        except Exception as e:
            print(f"Error during training: {e}")