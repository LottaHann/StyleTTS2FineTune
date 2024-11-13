# Use an official Ubuntu base image
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during the installation process
ENV DEBIAN_FRONTEND=noninteractive
# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1
# Set Flask environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    espeak-ng \
    python3.10 \
    python3-pip \
    python3-dev \
    bzip2 \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (to manage Python environments)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# Set the path for conda binaries
ENV PATH="/opt/conda/bin:${PATH}"

# Create the Conda environment
RUN conda create -n dataset python=3.10 -y

# Set up the working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt /app/

# Install dependencies in specific order to handle conflicts
SHELL ["/bin/bash", "-c"]
RUN source activate dataset && \
    # Install conda packages first
    conda install -y ipython jupyter && \
    # Install PyTorch ecosystem
    pip install --no-cache-dir torch==2.5.0+cu118 torchvision==0.20.0+cu118 torchaudio==2.5.0+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    # Install problematic packages separately
    pip install --no-cache-dir \
    asttokens==2.2.1 \
    executing==1.2.0 \
    pure-eval==0.2.2 \
    stack-data==0.6.2 && \
    # Install git repositories
    pip install --no-cache-dir \
    git+https://github.com/resemble-ai/monotonic_align.git@78b985be210a03d08bc3acc01c4df0442105366f \
    git+https://github.com/m-bain/whisperx.git@9e3a9e0e38fcec1304e1784381059a0e2c670be5 && \
    # Now install the requirements
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/temp /app/makeDataset/tools/audio /app/model/StyleTTS2/Data/wavs

# Copy the rest of the project files
COPY . /app

# Copy the .env file if it exists
COPY makeDataset/tools/.env /app/.env

# Ensure all directories have proper permissions
RUN chmod -R 755 /app

# Expose the port
EXPOSE 5000

# Add to Dockerfile environment settings
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Start Flask with proper logging
CMD ["conda", "run", "--no-capture-output", "-n", "dataset", "python", "-u", "app.py"]
