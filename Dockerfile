# Use an official Python runtime as a parent image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container first
COPY requirements.txt .

# Install the dependencies from requirements.txt, including the PyTorch index URL
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application code into the container
COPY . /app

# Make port 8002 available to the world outside this container
EXPOSE 8002

# Run the API using uvicorn when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]