FROM mambaorg/micromamba:1.4.9

# Create environment
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba install --yes --file /tmp/environment.yml
RUN micromamba clean --all --yes
