# Base Image
FROM nvidia/cuda:11.5.2-devel-ubuntu20.04

# Set Label
LABEL maintainer="Kailash Lohar <kailash@aganitha.ai>"

# Set Working Directory
WORKDIR /home

# Install Utilities & Miniconda Conda Environment
RUN apt-get update && apt-get install -y gcc g++ make wget curl ocl-icd-libopencl1 opencl-headers clinfo gnupg python3 python3-pip && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda && \
    rm -rf /var/lib/apt/lists/* && rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH /miniconda/bin:$PATH
COPY environment.yaml .
RUN conda env create -f environment.yaml && conda clean --all --yes
SHELL ["conda", "run", "-n", "vina_gpu", "/bin/bash", "-c"]

# Install Boost 1.77.0
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz \
    && tar -xzf boost_1_77_0.tar.gz && rm boost_1_77_0.tar.gz && cd /home/boost_1_77_0 && ./bootstrap.sh && ./b2 -d2 install

# Install NVIDIA Container Runtime
RUN curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | apt-key add - \
    && distribution=$(. /etc/os-release; echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | tee /etc/apt/sources.list.d/nvidia-container-runtime.list 
RUN mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Create a non-root user and switch to it
ARG USER=kailash
RUN useradd -m ${USER} && adduser ${USER} sudo
RUN chown -R ${USER}:${USER} /home
USER ${USER}

# Copy and Compile Vina-GPU+
COPY /Vina-GPU+ /home
RUN make source

WORKDIR /home/vina_gpu

# Expose Port for JupyterLab
EXPOSE 7777

# Set Default Command for starting JupyterLab
CMD ["conda", "run", "-n", "vina_gpu", "jupyter", "lab", "--ip=0.0.0.0", "--port=7777", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
