FROM python:3.8-slim

# Install necessary system dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    cmake \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for OpenBLAS
ENV BLAS=openblas \
    LAPACK=openblas

# Copy application files
COPY . /app

# Set the working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install numpy scipy pandas matplotlib scikit-learn catboost xgboost flask

# Set the command to run the application
CMD ["python", "app.py"]
