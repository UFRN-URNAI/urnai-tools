FROM alvarofpp/s2client:4.9.3 AS sc2client
FROM mambaorg/micromamba:2.3.1

# Create environment
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba install --yes --file /tmp/environment.yml
RUN micromamba clean --all --yes

# StarCraft 2
COPY --from=sc2client /root/StarCraftII /home/mambauser/StarCraftII
ENV SC2PATH=/home/mambauser/StarCraftII
# Fix:
#     WARNING:absl:SC2 isn't running, so bailing early on the websocket connection.
#     Failed to connect to the SC2 websocket. Is it up?
ENV LD_PRELOAD=""

# Project
COPY --chown=$MAMBA_USER:$MAMBA_USER . /home/mambauser/urnai
WORKDIR /home/mambauser/urnai

# User and packages
USER root
# hadolint ignore=DL3004,DL3008
RUN apt-get update -yq \
    && apt-get install --no-install-recommends -yq \
      sudo \
      vim \
      arandr \
    && echo "mambauser ALL=(ALL:ALL) NOPASSWD: ALL" >> /etc/sudoers \
    && sudo usermod -a -G root mambauser \
    && rm -rf /var/lib/apt/lists/*
USER mambauser

# Set user
ARG UID=1000
ARG GID=1000

USER root
RUN groupmod -g "${GID}" "${MAMBA_USER}"
RUN usermod -u "${UID}" -g "${GID}" "${MAMBA_USER}"
USER $MAMBA_USER

# Copy repository
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
WORKDIR /app
