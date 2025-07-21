from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading
import time

app = Flask(__name__)

# Global variables for model
model = None
tokenizer = None
model_loaded = False

def load_model():
    """Load the TinyLlama model in a separate thread with CPU-compatible settings"""
    global model, tokenizer, model_loaded
    print("Loading model...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    try:
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
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model_loaded = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle text generation, editing, and summarization requests"""
    global model, tokenizer, model_loaded
    
    if not model_loaded:
        return jsonify({'error': 'Model is still loading. Please wait...'}), 503
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data received'}), 400
    
    action = data.get('action', 'generate')
    user_message = data.get('message', '')
    context = data.get('context', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        if action == 'edit' and context:
            prompt = f"### Human: Edit the following text based on this instruction: {user_message}\n\nText: {context} ### Assistant:"
        elif action == 'summarize' and context:
            prompt = f"### Human: Summarize the following text: {context} ### Assistant:"
        else:
            prompt = f"### Human: {user_message} ### Assistant:"
        
        print(f"Processing prompt: {prompt}")  # Debug log
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True).to("cpu")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,
                use_cache=True
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Generated response: {response}")  # Debug log
        
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")  # Debug log
        return jsonify({'error': f'Error generating response: {str(e)}'}), 500

@app.route('/status')
def status():
    """Check model loading status"""
    return jsonify({'loaded': model_loaded})

if __name__ == '__main__':
    model_thread = threading.Thread(target=load_model)
    model_thread.start()
    
    print("Starting Flask server...")
    print("Model will load in the background.")
    print("Visit http://localhost:5000 to use the chatbot.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)