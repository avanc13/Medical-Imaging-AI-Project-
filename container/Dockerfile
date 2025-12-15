FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn9-runtime
WORKDIR /app
COPY container/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV PYTHONPATH=/app
CMD ["python", "inference/run_inference_n123.py", "--help"]

