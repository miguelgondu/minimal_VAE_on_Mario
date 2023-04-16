# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Update package list and install necessary dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common

# Install OpenJDK Java 17
RUN apt-get install -y openjdk-17-jdk

# Install Python 3.9 and pip
RUN apt-get install -y python3.9 python3.9-dev python3.9-venv python3-pip

# Set Python 3.9 as the default Python interpreter
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Set Java home environment variable
# ENV JAVA_HOME /usr/bin/java

# Set Python home environment variable
# ENV PYTHON_HOME /usr/bin/python3.9

# Set PATH to include Python and Java
# ENV PATH $JAVA_HOME/bin:$PYTHON_HOME:$PATH

# Install xvfb
RUN apt-get install -y xvfb

# Set working directory
WORKDIR /app

# Your additional steps here, e.g. copying application code to the container
COPY . /app/
WORKDIR /app
RUN pip install -r requirements.txt

# Set up a virtual screen using xvfb
ENV DISPLAY :99
RUN chmod a+x /app/start.sh

# Start the application
CMD ["./start.sh"]
