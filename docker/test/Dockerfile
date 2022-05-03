FROM python:3.6

# Install requirements
RUN pip3 install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Install URNAI
RUN pip3 install --no-cache-dir --no-dependencies git+https://github.com/UFRN-URNAI/urnai-tools@master

# Set environment variables
ENV WORKDIR=/app
WORKDIR ${WORKDIR}
