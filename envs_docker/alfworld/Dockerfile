FROM continuumio/miniconda3:latest

WORKDIR /opt/evals

RUN apt-get update && apt-get install -y \
    build-essential \
    make \
    gcc \
    cmake \
    zlib1g-dev \
    libffi-dev \
    libncurses5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml ./
COPY requirements.txt ./
RUN conda env create -f environment.yml

RUN conda run -n alfworld-env python -m pip install --upgrade pip && \
    conda run -n alfworld-env python -m pip install setuptools==68.2.2 wheel

RUN conda run -n alfworld-env python -m pip install --no-build-isolation -r requirements.txt
############################################################################################
RUN conda create -n alfworld-env python=3.11 pip -c conda-forge -y && \
    conda run -n alfworld-env pip install "setuptools==78.1.1" && \
    conda run -n alfworld-env pip install --no-build-isolation visdom==0.2.4 && \
    conda run -n alfworld-env pip install -r requirements.txt && \
    conda clean -afy

SHELL ["conda", "run", "-n", "alfworld-env", "/bin/bash", "-c"]

CMD ["bash"]

