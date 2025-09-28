# Use an official Python runtime as a parent image
FROM python:3.9-slim

# --- THIS IS THE FIX ---
# Install a more comprehensive set of system-level dependencies required by
# OpenCV and MediaPipe on headless Linux servers.
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 7860

# Define the command to run your app
CMD ["python", "app.py"]

