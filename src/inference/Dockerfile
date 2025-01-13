# # FROM python:3.9-slim

# # # Install dependencies
# # COPY requirements.txt .
# # RUN pip install --no-cache-dir -r requirements.txt

# # # Set up directories
# # WORKDIR /opt/ml/model
# # WORKDIR /opt/ml/code

# # # Copy inference script
# # COPY inference.py .

# # ENV PYTHONUNBUFFERED=TRUE
# # ENV PYTHONDONTWRITEBYTECODE=TRUE
# # ENV PATH="/opt/ml/code:${PATH}"

# # # Set entrypoint for SageMaker
# # ENTRYPOINT ["python", "/opt/ml/code/inference.py"]

# # FROM python:3.9-slim

# # # Install dependencies
# # COPY requirements.txt .
# # RUN pip install --no-cache-dir -r requirements.txt

# # # Set up directories
# # WORKDIR /opt/ml/model
# # WORKDIR /opt/ml/code

# # # Copy inference code
# # COPY inference.py /opt/ml/code/
# # COPY logger_config.py /opt/ml/code/
# # COPY monitoring.py /opt/ml/code/
# # COPY serve /opt/ml/code/
# # RUN chmod +x /opt/ml/code/serve

# # # Set environment variables
# # ENV PYTHONUNBUFFERED=TRUE
# # ENV PYTHONDONTWRITEBYTECODE=TRUE
# # ENV PATH="/opt/ml/code:${PATH}"

# # # Create log directory
# # RUN mkdir -p /opt/ml/model && \
# #     touch /opt/ml/model/inference.log && \
# #     chmod 666 /opt/ml/model/inference.log

# # # Set working directory for execution
# # WORKDIR /opt/ml/code

# # # Set entrypoint using python explicitly
# # ENTRYPOINT ["python", "/opt/ml/code/serve"]

# FROM python:3.9-slim

# # Install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Set up directories
# WORKDIR /opt/ml/model
# WORKDIR /opt/ml/code

# # Copy model artifacts
# COPY kmeans_model.joblib /opt/ml/model/
# COPY scaler.joblib /opt/ml/model/

# # Copy inference code
# COPY inference.py /opt/ml/code/
# COPY logger_config.py /opt/ml/code/
# COPY monitoring.py /opt/ml/code/
# COPY serve /opt/ml/code/
# RUN chmod +x /opt/ml/code/serve

# # Set environment variables
# ENV PYTHONUNBUFFERED=TRUE
# ENV PYTHONDONTWRITEBYTECODE=TRUE
# ENV PATH="/opt/ml/code:${PATH}"

# # Create log directory and set permissions
# RUN chmod -R 755 /opt/ml/model

# # Set working directory for execution
# WORKDIR /opt/ml/code

# # Set entrypoint using python explicitly
# ENTRYPOINT ["python", "/opt/ml/code/serve"]

FROM python:3.9-slim

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set up directories
WORKDIR /opt/ml/model
WORKDIR /opt/ml/code

# Copy inference code
COPY inference.py /opt/ml/code/
COPY logger_config.py /opt/ml/code/
COPY monitoring.py /opt/ml/code/
COPY serve /opt/ml/code/
RUN chmod +x /opt/ml/code/serve

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# Create log directory and set permissions
RUN mkdir -p /opt/ml/model && \
    touch /opt/ml/model/inference.log && \
    chmod -R 755 /opt/ml/model

# Set working directory for execution
WORKDIR /opt/ml/code

# Set entrypoint using python explicitly
ENTRYPOINT ["python", "/opt/ml/code/serve"]