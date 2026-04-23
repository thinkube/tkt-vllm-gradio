#!/usr/bin/env python3
"""vLLM Inference Server with Gradio UI"""

import os
import gradio as gr
from fastapi import FastAPI
import uvicorn
from vllm import LLM, SamplingParams

MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
APP_NAME = os.environ.get("APP_NAME", "vllm-server")
APP_TITLE = os.environ.get("APP_TITLE", APP_NAME)

app = FastAPI(title=f"{APP_NAME} vLLM Server")

print(f"Loading model with vLLM: {MODEL_ID}")
llm = LLM(model=MODEL_ID, dtype="auto")
tokenizer = llm.get_tokenizer()


def generate_response(message, history, temperature=0.7, max_tokens=512):
    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    outputs = llm.generate([prompt], sampling_params)
    response = outputs[0].outputs[0].text

    for i in range(0, len(response), 5):
        yield response[: i + 5]


demo = gr.ChatInterface(
    generate_response,
    type="messages",
    title=APP_TITLE,
    description=f"Chat with {MODEL_ID} (powered by vLLM)",
    examples=[
        {"text": "Hello! How are you?"},
        {"text": "Can you explain quantum computing in simple terms?"},
        {"text": "Write a Python function to calculate fibonacci numbers"},
    ],
    theme="soft",
    analytics_enabled=False,
    additional_inputs=[
        gr.Slider(0.1, 2.0, value=0.7, label="Temperature"),
        gr.Slider(64, 2048, value=512, label="Max Tokens"),
    ],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_ID, "engine": "vLLM"}


app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
