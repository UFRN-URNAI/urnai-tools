FROM mambaorg/micromamba:2.0.8

# Create environment
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba install --yes --file /tmp/environment.yml
RUN micromamba clean --all --yes
