# Use a lightweight and stable Python image based on Debian Bookworm
FROM python:3.12.10-slim-bookworm

# Copy the 'uv' and 'uvx' binaries directly from the official astral-sh image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory inside the container
WORKDIR /app

# Ensure installed packages from the virtual environment are available in PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy dependency definition files first to leverage Docker layer caching
COPY "pyproject.toml" "uv.lock" ".python-version" ./

# Install dependencies using uv (--locked ensures exact versions from uv.lock, 
# --no-dev skips development dependencies, --no-install-project skips the root package itself)
RUN uv sync --locked --no-dev --no-install-project

# Copy the prediction script and the serialized model artifact
COPY "predict.py" "model.bin" ./

# Clean up uv cache to reduce the final Docker image size
RUN uv cache clean

# Expose the port that the FastAPI application runs on
EXPOSE 9696

# Define the default command to start the application using Uvicorn
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]