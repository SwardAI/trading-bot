FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Don't run as root
RUN useradd -m botuser

# Create data directories and give botuser ownership
RUN mkdir -p data/historical data/logs && chown -R botuser:botuser data/

USER botuser

# Health check
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('data/bot.db') else 1)"

ENTRYPOINT ["python", "main.py"]
