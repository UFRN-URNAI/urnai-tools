FROM python:3.10-slim

# Virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install requirements
RUN pip3 install --upgrade pip
COPY requirements-dev.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Set environment variables
ENV WORKDIR=/app
WORKDIR ${WORKDIR}
