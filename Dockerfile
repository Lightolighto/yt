ARG PYTHON_VERSION=3.12.1
FROM python:${PYTHON_VERSION}-slim AS base

# Install system dependencies
RUN apt-get update && \
    apt-get install -y pipenv && \
    apt-get install -y \
    fontconfig \
    fonts-dejavu \
    fonts-dejavu-core \
    fonts-dejavu-extra \
    fonts-liberation \
    fonts-noto \
    git \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PORT=7860
ENV PIPENV_VENV_IN_PROJECT=1

# Create a non-root user and group
RUN groupadd -g 1000 appuser && \
    useradd -m -u 1000 -g appuser appuser

# Set the working directory and adjust permissions
WORKDIR /app
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Install dependencies (inside the virtual environment created by Pipenv)
RUN pipenv install --dev --ignore-pipfile && \
    pipenv run pip install fastapi asyncio uvicorn mutagen  requests imageio[ffmpeg] imageio[pyav] assemblyai moviepy git+https://github.com/yt-dlp/yt-dlp
# Expose the application port
EXPOSE 7860

# Command to run the application
CMD ["pipenv", "run", "python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "8", "--timeout-keep-alive", "600"]