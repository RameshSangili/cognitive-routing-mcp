FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PORT=8080

EXPOSE 8080

# uvicorn serves the ASGI app; $PORT is injected by Cloud Run at runtime
CMD ["sh", "-c", "uvicorn src.server:create_app --factory --host 0.0.0.0 --port $PORT"]
