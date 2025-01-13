# FROM python:3.9-slim

# # Install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Set up directories
# WORKDIR /opt/ml/model
# WORKDIR /opt/ml/code

# # Copy inference script
# COPY inference.py .

# ENV PYTHONUNBUFFERED=TRUE
# ENV PYTHONDONTWRITEBYTECODE=TRUE
# ENV PATH="/opt/ml/code:${PATH}"

# # Set entrypoint for SageMaker
# ENTRYPOINT ["python", "/opt/ml/code/inference.py"]


FROM python:3.9-slim

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set up directories
WORKDIR /opt/ml/model
WORKDIR /opt/ml/code

# Copy all inference code
COPY inference.py .
COPY model_handler.py .
COPY logger_config.py .
COPY monitoring.py .
COPY serve .
RUN chmod +x serve

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# Set entrypoint for SageMaker
ENTRYPOINT ["serve"]