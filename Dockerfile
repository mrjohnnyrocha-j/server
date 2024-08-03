# server/Dockerfile

FROM python:3.12-slim

ENV APP_HOME=/app
WORKDIR $APP_HOME

# Set up environment variables
ENV GROQ_API_KEY=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

# Copy and install dependencies
COPY pyproject.toml .
RUN pip install poetry && poetry install --no-dev

# Install additional packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Expose port for the application
EXPOSE 8000

# Run the application
CMD ["poetry", "run", "uvicorn", "app.main:app"]
