# Use an official Python runtime as a parent image
FROM python:3.7.3-alpine

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN rm -rf .git venv venv3.6

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py
ENV X_BEARER_TOKEN=""

# Run app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]
