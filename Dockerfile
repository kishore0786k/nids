FROM python:3.10-slim AS python-base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

FROM python-base AS api
COPY backend ./backend
COPY src ./src
COPY frontend ./frontend
COPY README.md CHANGELOG.md .env.example ./
RUN mkdir -p runs logs results data models
EXPOSE 5000
CMD ["python", "-m", "backend.app"]

FROM nginx:1.27-alpine AS frontend
COPY frontend/index.html /usr/share/nginx/html/index.html
COPY frontend/css /usr/share/nginx/html/static/css
COPY frontend/js /usr/share/nginx/html/static/js
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
