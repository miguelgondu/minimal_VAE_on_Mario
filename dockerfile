# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Update package list and install necessary dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get install -y openjdk-17-jdk && \
    apt-get install -y python3.9 python3.9-dev python3.9-venv python3-pip && \
    apt-get install -y xvfb

# Set Python 3.9 as the default Python interpreter
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Set working directory
WORKDIR /app

# Your additional steps here, e.g. copying application code to the container
COPY ./data /app/data
COPY ./mario_utils /app/mario_utils
COPY ./simulator.py /app/
COPY ./requirements_in_docker.txt /app/
COPY ./start.sh /app/
COPY ./vae.py /app/
COPY ./models /app/models/
COPY ./simulator.jar /app/

# Installing python dependencies
RUN pip install -r requirements_in_docker.txt -f https://download.pytorch.org/whl/torch_stable.html

# Set up a virtual screen using xvfb
ENV DISPLAY :99
RUN chmod a+x /app/start.sh

# Start the application
CMD ["/app/start.sh"]
