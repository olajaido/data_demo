FROM python:3.9-slim

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set up directories
WORKDIR /opt/ml/model
WORKDIR /opt/ml/code

# Copy inference script
COPY inference.py .

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# Set entrypoint for SageMaker
ENTRYPOINT ["python", "/opt/ml/code/inference.py"]