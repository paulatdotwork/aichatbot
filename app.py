from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

from transformers import AutoModelForCausalLM, AutoTokenizer

# Use GPT-Neo for generative tasks
model_name = "EleutherAI/gpt-neo-1.3B"  # You can also try gpt-neo-2.7B for better performance
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Your existing code continues...


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    temperature = data.get('temperature', 0.7)
    top_k = data.get('top_k', 40)
    top_p = data.get('top_p', 0.9)

    # Call the generate_text function with the new parameters
    response_text = generate_text(prompt, temperature=temperature, top_k=top_k, top_p=top_p)

    return jsonify({'response': response_text})

def generate_text(prompt, max_length=1000, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.2):
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Create an attention mask
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

    # Generate text with specified parameters
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            do_sample=True  # Enable sampling
        )

    # Decode and return the generated text
    return tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

if __name__ == '__main__':
    app.run(debug=True)
