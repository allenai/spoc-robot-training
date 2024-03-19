FROM ghcr.io/allenai/cuda:11.8-ubuntu20.04-v1.2.27 AS conda_env_builder

ENV APP_HOME /root/spoc
WORKDIR $APP_HOME

RUN conda create -n spoc -y python=3.10

RUN /opt/miniconda3/bin/conda install -n spoc -y -c conda-forge jsonnet ipython

FROM conda_env_builder AS requirements_installer

ENV PYTHON=/opt/miniconda3/envs/spoc/bin/python
ENV PIP=/opt/miniconda3/envs/spoc/bin/pip

COPY ./requirements.txt /root/requirements.txt

RUN ( \
    export PIP_SRC=/opt/miniconda3/envs/spoc/pipsrc; \
    $PIP install --no-cache-dir setuptools==66.0.0 \
    && $PIP install --no-cache-dir -r /root/requirements.txt \
)

RUN $PIP install --no-cache-dir --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+5d0ab8ab8760eb584c5ae659c2b2b951cab23246
RUN $PIP install --no-cache-dir torch==2.0.1 torchvision xformers lightning open_clip_torch

# Upgrading moviepy to ensure visualization doesn't break
RUN $PIP install --no-cache-dir --upgrade moviepy

# ffmpeg from conda is broken. We'll install it at a system level
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 nano -y

# Add espeak for phonemizer
RUN apt-get install espeak-ng -y

# Clean up conda to remove unused packages
RUN /opt/miniconda3/bin/conda clean -ya 

RUN $PYTHON -c "import nltk; nltk.download('wordnet'); nltk.download('wordnet2022')"

RUN touch /root/.git-credentials

# The -l flag makes bash act as a login shell and load /etc/profile, etc.
ENTRYPOINT ["bash", "-l"]