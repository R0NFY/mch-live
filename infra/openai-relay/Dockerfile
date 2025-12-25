FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY relay /app/relay

ENV PORT=3001
EXPOSE 3001

CMD ["sh", "-lc", "uvicorn relay.app:app --host 0.0.0.0 --port ${PORT}"]





