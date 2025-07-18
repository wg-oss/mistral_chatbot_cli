from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    print("Small Chatbot CLI (CPU-friendly)")
    print("Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("Type 'exit' to quit the chat.")

    # Use a much smaller model that works well on CPU
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Ensure proper tokenizer setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model (this may take a minute on CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu"
    )

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Exiting chat. Goodbye!")
            break
        if user_input == '':
            continue
        
        print("Generating response...", end="", flush=True)
        
        try:
            # Create prompt for TinyLlama
            prompt = f"<|system|>You are a helpful assistant.</s><|user|>{user_input}</s><|assistant|>"
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            print(f"\nAssistant: {response}\n")
            
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main() 