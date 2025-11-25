FROM python:3.11-slim

# Install system dependencies required for OpenCV and other libraries
# xvfb is required for Open3D visualization in a headless environment
# libgl1-mesa-glx is replaced by libgl1 in newer Debian versions
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
# We use pip to install directly from pyproject.toml or list them here
# Increased timeout for slow connections/emulation
RUN pip install --default-timeout=1000 --no-cache-dir streamlit
RUN pip install --default-timeout=1000 --no-cache-dir .

# Note: OpenMVS binaries are required. 
# In a real deployment, you would either:
# 1. Build OpenMVS in this Dockerfile (takes a long time)
# 2. Copy pre-built binaries from a build stage
# 3. Use a base image that already has OpenMVS
# For now, we assume the user might mount the binaries or they are included in the build context if present.

# Expose the port for Streamlit
EXPOSE 8501

# Run the application with xvfb-run to support headless rendering
CMD ["xvfb-run", "--auto-servernum", "--server-args='-screen 0 1024x768x24'", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]
