FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd -m -u 1000 user
COPY --chown=user:user app.py ./
USER user
ENV HF_HOME=/home/user/.cache/huggingface
EXPOSE 7860
CMD ["/bin/sh", "-c", "exec gunicorn --bind \"0.0.0.0:${PORT:-7860}\" --workers 1 --threads 4 --timeout 180 \"app:make_application()\""]
