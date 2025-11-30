import numpy as np
import json
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

# ================== Utility Functions ==================

def build_vocab(token_docs, min_freq=1):
    """Xây dựng vocab và thêm <UNK> - ĐÃ SỬA"""
    freq = defaultdict(int)
    for doc in token_docs:
        # ĐẢM BẢO doc là list của các từ, không phải string
        if isinstance(doc, str):
            words = doc.split()
        else:
            words = doc
            
        for w in words:
            # Lọc ra các từ thực sự (bỏ ký tự đặc biệt)
            if len(w) > 1 and w.isalpha():  # Chỉ lấy từ có ít nhất 2 chữ cái
                freq[w.lower()] += 1
    
    vocab_words = [w for w, c in sorted(freq.items()) if c >= min_freq]
    vocab = {w: i for i, w in enumerate(vocab_words)}
    vocab['<UNK>'] = len(vocab)
    reverse_vocab = {i: w for w, i in vocab.items()}
    
    print(f"Vocabulary size: {len(vocab)} (bao gồm <UNK>)")
    print(f"Sample REAL words: {list(vocab.keys())[:20]}")  # Kiểm tra
    return vocab, reverse_vocab

def sentence_embedding_avg(tokens, emb_matrix, vocab):
    """Embedding trung bình của câu"""
    vecs = []
    unk_idx = vocab.get('<UNK>')
    
    for w in tokens:
        idx = vocab.get(w, unk_idx)
        vecs.append(emb_matrix[idx])
    
    return np.mean(vecs, axis=0) if vecs else np.zeros(emb_matrix.shape[1])

def softmax(z):
    """Hàm softmax ổn định số học"""
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    """Hàm mất mát cross entropy"""
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))

# ================== Model Class ==================

class SentimentClassifier:
    def __init__(self, vocab_size, embed_dim=100, num_classes=2, lr=0.1):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.lr = lr
        
        self.embedding = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim))
        self.W = np.random.uniform(-0.1, 0.1, (embed_dim, num_classes))
        self.b = np.zeros(num_classes)

    def forward(self, tokens_list, vocab):
        """Forward pass"""
        embeddings = np.array([
            sentence_embedding_avg(tokens, self.embedding, vocab) 
            for tokens in tokens_list
        ])
        logits = embeddings @ self.W + self.b
        probs = softmax(logits)
        return embeddings, logits, probs

    def backward(self, embeddings, probs, y_true, tokens_list, vocab):
        """Backward pass"""
        batch_size = len(tokens_list)
        delta = (probs - y_true) / batch_size
        
        # Gradient cho weights và bias
        grad_W = embeddings.T @ delta
        grad_b = np.sum(delta, axis=0)
        grad_emb_avg = delta @ self.W.T
        
        # Cập nhật weights và bias
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        
        # Cập nhật embedding vectors
        for i, tokens in enumerate(tokens_list):
            if len(tokens) > 0:
                grad_per_word = grad_emb_avg[i] / len(tokens)
                for w in tokens:
                    idx = vocab.get(w)
                    if idx is not None:
                        self.embedding[idx] -= self.lr * grad_per_word

    def train(self, train_tokens, y_train, vocab, epochs=10, batch_size=64, validation_data=None, patience=3):
        """Huấn luyện model"""
        n_samples = len(train_tokens)
        
        # Chuyển đổi labels sang dạng số
        if isinstance(y_train[0], str):
            label_mapping = {'negative': 0, 'positive': 1}
            y_train_num = np.array([label_mapping[y] for y in y_train])
        else:
            y_train_num = np.array(y_train)
        
        # Chuẩn bị validation data nếu có
        if validation_data:
            x_val, y_val = validation_data
            if isinstance(y_val[0], str):
                y_val_num = np.array([label_mapping[y] for y in y_val])
            else:
                y_val_num = np.array(y_val)
            
        print(f"Training on {n_samples} samples for {epochs} epochs...")
        print(f"Batch size: {batch_size}, Learning rate: {self.lr}")
        
        # Theo dõi best loss cho early stopping
        best_loss = float('inf')
        patience_counter = 0
        training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            train_tokens_shuffled = [train_tokens[i] for i in indices]
            y_train_shuffled = y_train_num[indices]
            
            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = (n_samples + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                
                batch_tokens = train_tokens_shuffled[start:end]
                batch_y = y_train_shuffled[start:end]
                
                # One-hot encoding
                y_one_hot = np.zeros((len(batch_y), self.num_classes))
                y_one_hot[np.arange(len(batch_y)), batch_y] = 1
                
                # Forward và backward pass
                embeddings, _, probs = self.forward(batch_tokens, vocab)
                loss = cross_entropy_loss(y_one_hot, probs)
                
                # Tính accuracy cho batch
                batch_preds = np.argmax(probs, axis=1)
                batch_accuracy = np.mean(batch_preds == batch_y)
                
                epoch_loss += loss
                epoch_accuracy += batch_accuracy
                
                self.backward(embeddings, probs, y_one_hot, batch_tokens, vocab)
                
                # In progress mỗi 10% số batch
                if batch_idx % max(1, num_batches // 10) == 0:
                    print(f"  Batch {batch_idx+1}/{num_batches} - Loss: {loss:.4f}, Acc: {batch_accuracy:.4f}")
            
            # Tính giá trị trung bình cho epoch
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            training_history['loss'].append(avg_loss)
            training_history['accuracy'].append(avg_accuracy)

            # Đánh giá trên validation set nếu có
            val_metrics = ""
            if validation_data:
                val_preds, _ = self.predict(x_val, vocab)
                val_accuracy = np.mean(val_preds == y_val_num)
                
                # Tính validation loss
                val_embeddings, _, val_probs = self.forward(x_val, vocab)
                val_one_hot = np.zeros((len(y_val_num), self.num_classes))
                val_one_hot[np.arange(len(y_val_num)), y_val_num] = 1
                val_loss = cross_entropy_loss(val_one_hot, val_probs)
                
                training_history['val_accuracy'].append(val_accuracy)
                training_history['val_loss'].append(val_loss)
                val_metrics = f", Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Acc: {avg_accuracy:.4f}{val_metrics}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Lưu best model
                self.best_embedding = self.embedding.copy()
                self.best_W = self.W.copy()
                self.best_b = self.b.copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Khôi phục best model
                    self.embedding = self.best_embedding
                    self.W = self.best_W
                    self.b = self.best_b
                    break
        
        return training_history

    def plot_training_history(self, history):
        """Vẽ đồ thị loss và accuracy qua các epoch"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 4))
            
            # Plot loss
            plt.subplot(1, 2, 1)
            plt.plot(history['loss'], label='Training Loss')
            if 'val_loss' in history and history['val_loss']:
                plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot accuracy
            plt.subplot(1, 2, 2)
            plt.plot(history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in history and history['val_accuracy']:
                plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('training_history.png')
            plt.show()
            
        except ImportError:
            print("Matplotlib not available, skipping plots")
    
    def evaluate(self, x_test, y_test, vocab):
        """Đánh giá model trên test set"""
        if isinstance(y_test[0], str):
            label_mapping = {'negative': 0, 'positive': 1}
            y_test_num = np.array([label_mapping[y] for y in y_test])
        else:
            y_test_num = np.array(y_test)
        
        predictions, confidences = self.predict(x_test, vocab)
        accuracy = np.mean(predictions == y_test_num)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test_num, predictions)}")
        print(f"Classification Report:\n{classification_report(y_test_num, predictions, target_names=['Negative', 'Positive'])}")
        
        return accuracy
    
    def predict(self, tokens_list, vocab):
        """Dự đoán trên dữ liệu mới"""
        _, _, probs = self.forward(tokens_list, vocab)
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        return preds, confs

    def save(self, filepath):
        """Lưu model"""
        np.savez(
            filepath, 
            embedding=self.embedding, 
            W=self.W, 
            b=self.b,
            embed_dim=self.embed_dim, 
            vocab_size=self.vocab_size, 
            num_classes=self.num_classes
        )
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath):
        """Load model"""
        data = np.load(filepath, allow_pickle=True)
        model = SentimentClassifier(
            int(data['vocab_size']), 
            int(data['embed_dim']),
            int(data['num_classes'])
        )
        model.embedding = data['embedding']
        model.W = data['W']
        model.b = data['b']
        print(f"Model loaded from {filepath}")
        return model

# ================== Main Execution ==================

if __name__ == "__main__":
    print("Loading data...")
    data = np.load('data_tokens.npz', allow_pickle=True)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

    # DEBUG: Kiểm tra kiểu dữ liệu
    print(f"Type of x_train[0]: {type(x_train[0])}")
    print(f"Sample x_train[0]: {x_train[0][:100]}...")
    
    # CHUYỂN ĐỔI DỮ LIỆU NẾU CẦN
    if isinstance(x_train[0], str):
        print("Converting string documents to word lists...")
        x_train = [doc.split() for doc in x_train]
        x_test = [doc.split() for doc in x_test]
    
    print(f"After conversion - Type of x_train[0]: {type(x_train[0])}")
    print(f"First 10 words of x_train[0]: {x_train[0][:10]}")

    print("Building vocabulary...")
    vocab, reverse_vocab = build_vocab(x_train, min_freq=2)
    
    # Chuyển đổi labels
    label_mapping = {'negative': 0, 'positive': 1}
    if isinstance(y_train[0], str):
        y_train_num = np.array([label_mapping[y] for y in y_train])
        y_test_num = np.array([label_mapping[y] for y in y_test])
    else:
        y_train_num = np.array(y_train)
        y_test_num = np.array(y_test)

    print(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")
    
    # Chỉ tiếp tục nếu vocabulary có ý nghĩa
    if len(vocab) > 1000:
        # Tách validation set từ training data
        split_idx = int(0.8 * len(x_train))
        x_train_split = x_train[:split_idx]
        y_train_split = y_train_num[:split_idx]
        x_val = x_train[split_idx:]
        y_val = y_train_num[split_idx:]
        
        model = SentimentClassifier(
            vocab_size=len(vocab), 
            embed_dim=100, 
            num_classes=2, 
            lr=0.1
        )
        
        # Train với validation data - CHỈ MỘT LẦN
        history = model.train(
            x_train_split, 
            y_train_split, 
            vocab, 
            epochs=20, 
            batch_size=64,
            validation_data=(x_val, y_val),
            patience=5
        )
        
        # Vẽ biểu đồ training history
        model.plot_training_history(history)
        
        # Đánh giá trên test set
        model.evaluate(x_test, y_test, vocab)
        
        # Lưu model
        model.save('sentiment_model.npz')
        
        # Lưu vocab
        with open('vocab.json', 'w') as f:
            json.dump(vocab, f)
    else:
        print(f"❌ Vocabulary quá nhỏ ({len(vocab)} từ). Kiểm tra lại dữ liệu đầu vào!")