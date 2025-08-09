# Multi-stage Dockerfile for BudSimulator
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_16.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install llm-memory-calculator
COPY llm-memory-calculator /app/llm-memory-calculator
RUN pip install --no-cache-dir -e /app/llm-memory-calculator

# Copy BudSimulator requirements and install Python dependencies
COPY BudSimulator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy BudSimulator application
COPY BudSimulator /app/BudSimulator
WORKDIR /app/BudSimulator

# Install frontend dependencies including dev dependencies
RUN cd frontend && npm install --include=dev

# Build frontend
RUN cd frontend && npm run build

# Create non-root user
RUN useradd -m -u 1000 user
RUN chown -R user:user /app/BudSimulator/frontend/node_modules
USER user

# Expose ports
EXPOSE 8000 3000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start both backend and frontend
CMD ["sh", "-c", "python run_api.py & cd frontend && npm start"]