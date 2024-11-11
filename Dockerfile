# Use an official Ubuntu base image
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during the installation process
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    espeak-ng \
    python3.10 \
    python3-pip \
    python3-dev \
    bzip2 \
    ca-certificates \
    && apt-get clean

# Install Miniconda (to manage Python environments)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# Set the path for conda binaries and activate conda in all shell sessions
ENV PATH="/opt/conda/bin:${PATH}"

# Create the Conda environment and activate it
RUN conda create --name dataset python=3.10 && \
    conda clean -afy

# Activate the Conda environment and install dependencies
RUN echo "source activate dataset" > ~/.bashrc
SHELL ["conda", "run", "-n", "dataset", "/bin/bash", "-c"]

# Install PyTorch and other dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -U

# Install whisperx, phonemizer, pydub, pysrt, and tqdm
RUN pip install git+https://github.com/m-bain/whisperx.git phonemizer pydub pysrt tqdm

# Set up the working directory
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Ensure the .env file is available for the app
COPY makeDataset/tools/.env /app/.env

# Expose the port (if applicable)
EXPOSE 5000

# Run the app (replace with the command that starts your app)
CMD ["conda", "run", "-n", "dataset", "python", "your_app.py"]
