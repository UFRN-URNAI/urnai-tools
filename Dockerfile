FROM tensorflow/tensorflow:2.6.1-gpu-jupyter

# Install requirements
RUN pip3 install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Set environment variables
ENV WORKDIR=/app
WORKDIR ${WORKDIR}
