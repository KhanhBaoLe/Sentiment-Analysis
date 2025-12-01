import os 

print("Current directoryL", os.getcwd())
print("\nFiles in current directory:")
for file in os.listdir('.'):
    if file.endswith('.h5') or file.endswith('.json') or file.endswith('.npz'):
        print(f"📁 {file}")
        
# Check specifically for DL model files
print("\nChecking for DL model files:")
print("dl_model.h5 exists:", os.path.exists('dl_model.h5'))
print("dl_tokenizer.json exists:", os.path.exists('dl_tokenizer.json'))
print("sentiment_model.nps exists:", os.path.exists('sentiment_model.npz'))
print("vocab.json exists:", os.path.exists('vocab.json'))