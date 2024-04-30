# Use an official Python runtime as a parent image
FROM python:3.7.3

# Set the working directory in the container
WORKDIR /app

# Copy just the requirements.txt first to leverage Docker cache
COPY requirements.txt /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . /app

# Remove unnecessary files if needed
RUN rm -rf .git venv venv3.6

EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py
ENV X_BEARER_TOKEN=""

# Run app.py when the container launches
CMD ["python", "app.py"]
