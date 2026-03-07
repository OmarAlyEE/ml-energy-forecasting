
# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]
