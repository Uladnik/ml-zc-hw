FROM python:3.12.10-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
COPY "pyproject.toml" "uv.lock" ".python-version" ./
COPY "src/" ./src/
RUN uv sync --locked --no-dev --no-install-project
COPY "predict.py" "model.bin" ./
EXPOSE 9696
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]