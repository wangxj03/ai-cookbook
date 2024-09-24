FROM node:current as builder

# Clone the demo-code-search repository to use the frontend code
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/qdrant/demo-code-search.git

WORKDIR /demo-code-search/frontend

RUN npm install; npm run build

# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the Python files into the container
COPY backend ./backend

# Copy the frontend files into the container
COPY --from=builder /demo-code-search/frontend/dist ./frontend/dist

ENTRYPOINT ["python", "-m", "backend.service"]
