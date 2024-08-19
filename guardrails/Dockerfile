# Use an official Python runtime as the base image
FROM python:3.11-slim

# Install necessary packages
RUN apt-get update && apt-get install -y iputils-ping curl

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python files into the container
COPY genai/ ./genai

# Expose the port the app runs on
EXPOSE 8000

# Run the application
ENTRYPOINT ["python", "-m", "genai.app"]
