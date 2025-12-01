import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def test_calibration():
    model = tf.keras.models.load_model('dl_model.h5')
    with open('dl_tokenizer.json', 'r') as f:
        tokenizer_data = json.load(f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)
    
    test_texts = [
        "this film is good and amazing",
        "this movie is terrible and awful", 
        "great excellent wonderful fantastic",
        "bad horrible terrible awful",
        "I love this film",
        "I hate this movie"
    ]
    
    print("Calibration Test:")
    for text in test_texts:
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=100)
        prediction = model.predict(padded_sequence, verbose=0)[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        print(f"'{text}' -> {sentiment} ({prediction:.4f})")

if __name__ == "__main__":
    test_calibration()