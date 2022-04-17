FROM python:3.6

# Install requirements
RUN pip3 install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Set environment variables
ENV WORKDIR=/app
WORKDIR ${WORKDIR}
