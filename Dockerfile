FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && apt install -y \
    python3-pip python3-dev \
    git curl wget build-essential cmake pkg-config \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx \
    ros-humble-cv-bridge ros-humble-vision-msgs \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install your exact Python dependencies
RUN pip install \
    numpy==1.26.4 \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html \
    opencv-python==4.10.0.84 \
    loguru==0.7.2 \
    scikit-image==0.24.0 \
    tqdm==4.66.5 \
    Pillow==10.4.0 \
    thop>=0.1.1 \
    ninja==1.11.1.1 \
    tabulate==0.9.0 \
    tensorboard==2.18.0 \
    lap==0.4.0 \
    motmetrics==1.4.0 \
    filterpy==1.4.5 \
    h5py==3.12.1 \
    cython==3.0.11 \
    setuptools==59.8.0 \
    cython_bbox==0.1.5 \
    pycocotools==2.0.8 \
    torchsummary \
    onnx==1.17.0 \
    onnxruntime==1.19.2 \
    onnx-simplifier==0.4.36

# Setup ROS workspace
RUN mkdir -p /ros2_ws/src
WORKDIR /
RUN git clone https://github.com/azzy13/vehicle_perception.git
WORKDIR /ros2_ws



CMD ["bash"]
