from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import threading
import time

app = Flask(__name__)

# Global variables for model
model = None
tokenizer = None
model_loaded = False

def load_model():
    """Load the model in a separate thread"""
    global model, tokenizer, model_loaded
    
    print("Loading model...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    model.eval()
    model_loaded = True
    print("Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global model, tokenizer, model_loaded
    
    if not model_loaded:
        return jsonify({'error': 'Model is still loading. Please wait...'})
    
    data = request.get_json()
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'})
    
    try:
        # Create prompt for TinyLlama
        prompt = f"<|system|>You are a helpful assistant.</s><|user|>{user_message}</s><|assistant|>"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,
                use_cache=True
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': f'Error generating response: {str(e)}'})

@app.route('/status')
def status():
    return jsonify({'loaded': model_loaded})

if __name__ == '__main__':
    # Load model in background thread
    model_thread = threading.Thread(target=load_model)
    model_thread.start()
    
    print("Starting Flask server...")
    print("Model will load in the background.")
    print("Visit http://localhost:5000 to use the chatbot.")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 