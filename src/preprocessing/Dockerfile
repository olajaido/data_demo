FROM python:3.9-slim

WORKDIR /opt/ml/processing

# Install required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy preprocessing script
COPY preprocessing.py .

ENTRYPOINT ["python", "preprocessing.py"]