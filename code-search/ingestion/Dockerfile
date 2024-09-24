# Use an official Python runtime as the base image
FROM python:3.11-slim

# Clone the qdrant repository
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/qdrant/qdrant.git

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the Python files into the container
COPY jobs ./jobs

COPY run.sh .

ENTRYPOINT ["./run.sh"]
