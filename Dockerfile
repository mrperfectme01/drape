FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY drape ./drape
COPY scripts ./scripts
# Build data, models and the serving engine at image build time (~15 min on 2 cores)
RUN python -m scripts.train
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "drape.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
