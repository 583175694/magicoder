from transformers import AutoTokenizer
import transformers
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

base_model = "ise-uiuc/Magicoder-S-DS-6.7B"
device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained(base_model)
pipeline = transformers.pipeline(
    "text-generation",
    model=base_model,
    torch_dtype=torch.float16,
    device=device
)


def evaluate_magicoder(instruction, temperature=1, max_new_tokens=2048):
    MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

    @@ Instruction
    {instruction}

    @@ Response
    """ 
    
    prompt = MAGICODER_PROMPT.format(instruction=instruction)

    if temperature > 0:
        sequences = pipeline(
            prompt,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    else:
        sequences = pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
        )
    generated_text = sequences[0]['generated_text'].replace(prompt, "")
    return generated_text


@app.route("/magicoder", methods=["POST"])
def magicoder_api():
    data = request.get_json()
    instruction = data.get("instruction", "")
    temperature = data.get("temperature", 1)
    max_new_tokens = data.get("max_new_tokens", 2048)

    result = evaluate_magicoder(instruction, temperature, max_new_tokens)

    return jsonify({"response": result})


if __name__ == "__main__":
    app.run(port=8080)
