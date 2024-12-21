from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig

app = Flask(__name__)

# Load your model and tokenizer
def load_model():
    model_path = "./merged_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="left")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True)
    )
    return model, tokenizer

model, tokenizer = load_model()

def generate_response(message):
    # Format the input and generate the response (simplified)
    inputs = tokenizer(message, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

@app.route("/sms", methods=["POST"])
def sms_reply():
    # Get the incoming message
    incoming_msg = request.form.get("Body", "")
    # Generate chatbot response
    response_text = generate_response(incoming_msg)
    # Create a Twilio response
    response = MessagingResponse()
    response.message(response_text)
    return str(response)

if __name__ == "__main__":
    app.run(debug=True)
