FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY config ./config
COPY evals ./evals
ENV DATA_DIR=/app/data LOG_DIR=/app/logs KOMMUN_CONFIG=config/kommun.yaml
EXPOSE 8000
CMD ["kommun-agent", "serve", "--port", "8000"]
