# Use an official Python image as a base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies

RUN pip install --upgrade pip \
    && pip install --no-cache-dir pipenv \
    && pipenv --python 3.10 \
    && pipenv install --deploy --ignore-pipfile



# Set a default environment variable (can be overridden at runtime)
ENV ENABLE_LOGGING=True

# Default command to run the script with a task argument
ENTRYPOINT ["pipenv", "run", "python", "-m", "apple_stock_extremes_ml"]