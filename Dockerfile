# Use an official Python runtime as a parent image
# FROM python:3.11.7
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose port 5635 to the world outside this container
EXPOSE 5637

# Command to run the FastAPI app with Uvicorn on port 5635
CMD ["uvicorn", "mgov-apiv4-rest:app", "--host", "0.0.0.0", "--port", "5637", "--workers", "4"]
