TinyLlama Text Generation Chatbot
A Flask-based web application for text generation and editing using the TinyLlama-1.1B-Chat-v1.0 model from Hugging Face Transformers.
Features

Generate text based on user prompts (e.g., write articles, emails, or stories).
Edit existing text (e.g., rewrite, summarize, or change tone).
Interactive web-based chat interface for conversational AI.
Optimized model loading with quantization for efficient inference.

Requirements

Python 3.8+
Install dependencies:pip install -r requirements.txt



Usage

Clone the repository:git clone <repository-url>
cd <repository-directory>


Install dependencies:pip install -r requirements.txt


Run the application:python app.py


Visit http://localhost:5000 to use the chatbot.

Model

Uses: TinyLlama/TinyLlama-1.1B-Chat-v1.0 from Hugging Face Transformers.
Supports text generation, editing, and summarization tasks.

Fine-Tuning
To improve performance for text generation and editing:

Prepare a dataset with instruction-response pairs (e.g., JSONL format with prompts and responses for writing/editing).
Fine-tune using LoRA with the peft library:python fine_tune.py

(See fine_tune.py for an example script, not included in this repository.)

Notes

The model loads in the background to reduce startup time.
GPU support is enabled if available; otherwise, it falls back to CPU.
Use the web interface to select actions (generate, edit, summarize) and provide context for editing tasks.
