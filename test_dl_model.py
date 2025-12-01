import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def test_dl_model():
    try:
        # load model
        model = tf.keras.models.load_model('dl_model.h5')
        print("✅ Model loaded successfully!")
        
        # Load tokenizer
        with open('dl_tokenizer.json', 'r') as f:
            tokenizer_data = json.load(f)
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)
        print("✅ Tokenizer loaded successfully!")
        
        # Test prediction
        test_text = "this film is good"
        sequence = tokenizer.texts_to_sequences([test_text])
        padded_sequence = pad_sequences(sequence, maxlen=100)
        
        prediction = model.predict(padded_sequence, verbose=0)[0][0]
        
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        
        print(f"✅ Test prediction: {sentiment} (confidence: {prediction:.4f})")
        return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
if __name__ == "__main__":
    test_dl_model()