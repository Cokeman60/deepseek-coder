
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

import gradio as gr

def infer(prompt):
    result = generator(prompt, max_new_tokens=512, do_sample=False)[0]['generated_text']
    return result

gr.Interface(fn=infer, inputs="textbox", outputs="textbox", title="DeepSeek Coder 6.7B (CPU)").launch(server_port=7860)
