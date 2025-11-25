# Use a slim Python base image
FROM python:3.10-slim

# Ensure stdout/stderr are unbuffered
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Workdir inside the container
WORKDIR /app

# System deps (if you need any later, e.g. build tools, add here)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential && \
#     rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code (including data/raw_pdfs)
COPY . .

# Build the processed brochures dataset at build time
# Assumes your PDFs are under data/raw_pdfs in the repo
RUN python -m travelai.data_ingestion

# Expose FastAPI port
EXPOSE 8000

# Default command: run the API with Uvicorn
# OPENAI_API_KEY must be provided at runtime as an environment variable
CMD ["uvicorn", "travelai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
