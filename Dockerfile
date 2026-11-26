FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY detector.py app.py ./
COPY artifacts/detector.json artifacts/metrics.json ./artifacts/
COPY templates/ ./templates/
COPY static/ ./static/
USER 65532:65532
ENV APP_HOST=0.0.0.0 PORT=8080
EXPOSE 8080
CMD ["python", "app.py"]
