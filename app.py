from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import threading
import time
import re

app = Flask(__name__)

# Global variables for model
model = None
tokenizer = None
model_loaded = False

class CustomStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria to prevent repetitive generation"""
    def __init__(self, tokenizer, stop_sequences=None):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences or ["### Human:", "Human:", "Assistant:", "### Assistant:"]
        
    def __call__(self, input_ids, scores, **kwargs):
        # Convert current generation to text
        current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Check for stop sequences
        for stop_seq in self.stop_sequences:
            if stop_seq in current_text:
                return True
                
        # Check for repetitive patterns (same sentence repeated)
        sentences = current_text.split('.')
        if len(sentences) > 3:
            last_three = [s.strip().lower() for s in sentences[-4:-1] if s.strip()]
            if len(set(last_three)) == 1 and len(last_three) == 3:
                return True
                
        return False

def clean_response(text):
    """Clean up the generated response"""
    # Remove any remaining prompt artifacts
    text = re.sub(r'### Human:.*?### Assistant:', '', text, flags=re.DOTALL)
    text = re.sub(r'Human:.*?Assistant:', '', text, flags=re.DOTALL)
    
    # Remove repetitive patterns
    sentences = text.split('.')
    cleaned_sentences = []
    seen_sentences = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence.lower() not in seen_sentences:
            cleaned_sentences.append(sentence)
            seen_sentences.add(sentence.lower())
        elif sentence and len(cleaned_sentences) > 0:
            # Stop if we encounter a repeated sentence
            break
    
    # Reconstruct text
    result = '. '.join(cleaned_sentences)
    if result and not result.endswith('.'):
        result += '.'
    
    # Remove any trailing incomplete sentences
    result = re.sub(r'\.\s*[A-Z][^.]*$', '.', result)
    
    return result.strip()

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
        # Improved prompt formatting
        if action == 'edit' and context:
            prompt = (
                "<|im_start|>user\n"
                "You are an AI editor. Edit the following text according to the instruction, "
                "but keep all other content unchanged unless the instruction says to remove or replace it. "
                "If the instruction is to continue or add, append your response to the end.\n\n"
                f"Text:\n{context}\n\nInstruction: {user_message}<|im_end|>\n<|im_start|>assistant\n"
            )
        elif action == 'summarize' and context:
            prompt = f"<|im_start|>user\nSummarize this text: {context}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        
        print(f"Processing prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True).to("cpu")
        
        # Custom stopping criteria
        stopping_criteria = StoppingCriteriaList([
            CustomStoppingCriteria(tokenizer)
        ])
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=200,  # Reduced to prevent long rambling
                min_new_tokens=10,   # Ensure some response
                temperature=0.5,     # Lower temperature for more focused responses
                top_p=0.8,          # Slightly lower for less randomness
                top_k=40,           # Add top-k sampling
                do_sample=True,
                repetition_penalty=1.2,  # Penalize repetition
                length_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,
                use_cache=True,
                stopping_criteria=stopping_criteria
            )
        
        # Decode only the new tokens
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Clean up the response
        response = clean_response(response)
        
        print(f"Generated response: {response}")
        
        # Fallback for empty responses
        if not response.strip():
            response = "I apologize, but I'm having trouble generating a response right now. Could you try rephrasing your question?"
        
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': f'Error generating response: {str(e)}'}), 500

@app.route('/status')
def status():
    """Check model loading status"""
    return jsonify({'loaded': model_loaded})

# In-memory storage for a single project
projects = {
    'project_1': {
        'id': 'project_1',
        'title': 'Project 1',
        'content': '',
        'created_at': time.time(),
        'updated_at': time.time()
    }
}

@app.route('/projects', methods=['GET'])
def get_projects():
    """Get the single project"""
    return jsonify({'projects': [projects['project_1']]})

@app.route('/projects', methods=['POST'])
def create_project():
    """Always return/create the single project_1"""
    data = request.get_json() or {}
    project = projects['project_1']
    # Optionally update title/content if provided
    if 'title' in data:
        project['title'] = data['title']
    if 'content' in data:
        project['content'] = data['content']
    project['updated_at'] = time.time()
    return jsonify({'project': project})

@app.route('/projects/<project_id>', methods=['GET'])
def get_project(project_id):
    """Get the single project_1 regardless of id"""
    project = projects['project_1']
    return jsonify({'project': project})

@app.route('/projects/<project_id>', methods=['PUT'])
def update_project(project_id):
    """Update the single project_1 regardless of id"""
    data = request.get_json() or {}
    project = projects['project_1']
    if 'title' in data:
        project['title'] = data['title']
    if 'content' in data:
        project['content'] = data['content']
    project['updated_at'] = time.time()
    return jsonify({'project': project})

@app.route('/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Reset the single project_1's content (never actually delete)"""
    project = projects['project_1']
    project['content'] = ''
    project['updated_at'] = time.time()
    return jsonify({'success': True})

if __name__ == '__main__':
    model_thread = threading.Thread(target=load_model)
    model_thread.start()
    
    print("Starting Flask server...")
    print("Model will load in the background.")
    print("Visit http://localhost:5000 to use the chatbot.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)