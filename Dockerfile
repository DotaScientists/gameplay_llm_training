FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --yes --quiet \
    && apt-get install software-properties-common --yes --quiet \
    && add-apt-repository --yes ppa:deadsnakes/ppa \
    && apt-get update --yes --quiet

RUN apt-get install --yes --quiet --no-install-recommends python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    curl \
    build-essential \
    ca-certificates

RUN rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 999 \
    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python


RUN adduser --home /app app
USER app
WORKDIR /app

RUN python -m venv env
ENV PATH="/app/env/bin:$PATH"

WORKDIR /app/gameplay_llm_training

COPY --chown=app requirements.txt pip.ini pyproject.toml /app/gameplay_llm_training/

ENV PIP_CONFIG_FILE=/app/gameplay_llm_training/pip.ini


RUN /app/env/bin/pip install -r requirements.txt --no-cache-dir


COPY --chown=app README.md /app/gameplay_llm_training/README.md
COPY --chown=app src /app/gameplay_llm_training/src/
RUN /app/env/bin/pip install -e . --no-deps



CMD ["/app/env/bin/python", "-m", "gameplay_llm_training"]
