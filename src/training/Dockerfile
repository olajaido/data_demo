FROM python:3.9-slim

WORKDIR /opt/ml/code

# Install required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY train.py .

ENTRYPOINT ["python", "train.py"]