# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependency files
COPY requirements.txt /app/

# Install git, which is required for installing dependencies from git
RUN apt-get update && apt-get install -y --no-install-recommends git

# Install the application dependencies
RUN pip install -r requirements.txt

# Copy the model download script and download the model
# This is done in a separate layer to improve caching
COPY download_model.py /app/
RUN python download_model.py

# Copy the application source code
COPY . /app/

# Set the entrypoint for the container
ENTRYPOINT ["python", "stag.py"]

# Set a default command to show usage
CMD ["--help"]
