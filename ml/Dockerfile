FROM nvidia/cuda:10.1-base-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    wget \
    zip \
    sudo \
    ca-certificates \
    swig \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# Create project directory
ENV HOME=/home/user
RUN chmod 777 /home/user
RUN mkdir -p $HOME/pose-estimation
WORKDIR $HOME/pose-estimation

# Install python3 with pyenv
ENV PYENV_ROOT $HOME/.pyenv
ENV PYTHON_VERSION miniconda3-4.3.30
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv
RUN pyenv install $PYTHON_VERSION
RUN pyenv global $PYTHON_VERSION

# Install dependencies for openpose
RUN git clone https://github.com/ildoonet/tf-pose-estimation openpose
WORKDIR openpose
RUN conda install -y opencv
RUN pip install Cython tensorflow tensorflow-gpu
RUN pip install -r requirements.txt
RUN cd models/graph/cmu && \
    bash download.sh
RUN cd tf_pose/pafprocess/ && \
    swig -python -c++ pafprocess.i && \
    python3 setup.py build_ext --inplace

# Start ML server
WORKDIR $HOME/pose-estimation
ENV PYTHONPATH "${PYTHONPATH}:openpose"
ADD requirements.txt .
ADD server.py .
RUN pip install -r requirements.txt

ADD classifier classifier
RUN pip install -r classifier/requirements.txt

CMD ["python", "server.py"]
