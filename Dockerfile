# Use a slim, official Python image as a base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary files for the app to run
# We explicitly copy *only* what's needed for inference
COPY app.py .
COPY params.yaml .
COPY model /app/model/
COPY artifacts /app/artifacts/

# Expose the port that gunicorn will run on
EXPOSE 5000

# Set the command to run the application using gunicorn
# This is a production-grade server
# It runs 4 worker processes to handle requests
CMD ["gunicorn", "--workers=4", "--bind", "0.0.0.0:5000", "app:app"]