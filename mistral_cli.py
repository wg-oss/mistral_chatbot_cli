import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    print("Mistral 7B Chatbot CLI")
    print("Model: mistralai/Mixtral-8x7B-Instruct-v0.1")
    print("Type 'exit' to quit the chat.")

    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Ensure proper tokenizer setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Exiting chat. Goodbye!")
            break
        if user_input == '':
            continue
        
        # Create a simple prompt
        prompt = f"<s>[INST] {user_input} [/INST]"
        
        print("Generating response...", end="", flush=True)
        
        try:
            # Tokenize with explicit attention mask
            encoded = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response (only new tokens)
            response_ids = outputs[0][input_ids.shape[1]:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            
            print(f"\nAssistant: {response}\n")
            
        except Exception as e:
            print(f"\nError: {e}")
            print("Trying alternative approach...")
            
            # Fallback: simpler approach
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=64,
                        temperature=0.7,
                        do_sample=True
                    )
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                print(f"\nAssistant: {response}\n")
            except Exception as e2:
                print(f"\nFailed to generate response: {e2}")

if __name__ == "__main__":
    main() 