# Dockerfile for zarr file based analyzer
# Based on Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt update && apt install -y \
    clinfo \
    pocl-opencl-icd \
    ocl-icd-opencl-dev \
    ocl-icd-libopencl1

# Install Package dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /workspace

COPY cell_analyzer.py .
COPY vessel_analyzer.py .