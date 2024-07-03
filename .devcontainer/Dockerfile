FROM python:3.10.13-slim

# Install sudo and create the vscode user
RUN apt-get update && apt-get install -y sudo && rm -rf /var/lib/apt/lists/* \
    && useradd -ms /bin/bash vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Use sudo for commands requiring root privileges
# Install Git and necessary system libraries
RUN sudo apt-get update && sudo apt-get install -y \
    git \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libglib2.0-dev \
    && sudo rm -rf /var/lib/apt/lists/
    
# Switch to the vscode user
USER vscode
WORKDIR /app

# Copy only requirements.txt to leverage Docker cache
COPY --chown=vscode:vscode requirements.txt /app/requirements.txt

# Install Python dependencies
RUN python -m pip install --upgrade pip && python -m pip install -r /app/requirements.txt

# Install imageio-ffmpeg separately to avoid conflicts
RUN python -m pip install imageio-ffmpeg==0.4.8

# Copy the rest of the application code
COPY --chown=vscode:vscode . /app