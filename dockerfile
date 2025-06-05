
FROM python:3.10-slim

RUN apt update && apt install -y git &&     pip install torch transformers accelerate gradio sentencepiece

ENV MODEL_ID=deepseek-ai/deepseek-coder-6.7b-instruct

COPY app.py /app/app.py
WORKDIR /app

CMD ["python3", "app.py"]
