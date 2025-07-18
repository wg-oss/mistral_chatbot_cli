# Mistral 7B Tokenizer CLI

This command-line tool allows you to compress (tokenize) and reconstruct (decode) non-AI-generated text using the Mistral 7B model's tokenizer from Hugging Face Transformers.

## Features
- Tokenize (compress) any input text to token IDs
- Reconstruct (decode) text from token IDs
- Interactive command-line interface

## Requirements
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage
Run the script:
```bash
python mistral_cli.py
```

### Commands
- Enter any text to see its token IDs (compression)
- Type `reconstruct` to decode the last token IDs back to text
- Type `exit` to quit

## Model
- Uses: `mistralai/Mixtral-8x7B-Instruct-v0.1` tokenizer from Hugging Face

---
Inspired by the original Mistral code, adapted for Hugging Face Transformers. 