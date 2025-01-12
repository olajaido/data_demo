# Provider configuration (unchanged)
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = "eu-west-2"
}

# VPC and Networking (unchanged)
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "retail-analysis-vpc"
  }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "eu-west-2a"
  map_public_ip_on_launch = true

  tags = {
    Name = "retail-analysis-public"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "retail-analysis-igw"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "retail-analysis-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# S3 Configuration (unchanged)
resource "aws_s3_bucket" "retail_data_lake" {
  bucket = "retail-analysis-data-demo"

  tags = {
    Environment = "Production"
    Project     = "RetailAnalysis"
  }
}

resource "aws_s3_bucket_versioning" "retail_data_versioning" {
  bucket = aws_s3_bucket.retail_data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

# SageMaker Processing Job
resource "aws_sagemaker_processing_job" "retail_preprocessing" {
  name     = "retail-data-preprocessing"
  role_arn = aws_iam_role.sagemaker_role.arn

  processing_resources {
    cluster_config {
      instance_count    = 1
      instance_type     = "ml.m5.xlarge"
      volume_size_in_gb = 30
    }
  }

  input_config {
    input_name = "retail-input"
    s3_input {
      s3_uri        = "${aws_s3_bucket.retail_data_lake.bucket}/online_retail_II.xlsx"
      local_path    = "/opt/ml/processing/input"
      s3_data_type  = "S3Prefix"
      s3_input_mode = "File"
    }
  }

  output_config {
    output_name = "retail-processed"
    s3_output {
      s3_uri     = "${aws_s3_bucket.retail_data_lake.bucket}/processed"
      local_path = "/opt/ml/processing/output"
    }
  }

  app_specification {
    image_uri = "${aws_ecr_repository.retail_models.repository_url}:latest"
    container_arguments = [
      "--input-data", "/opt/ml/processing/input",
      "--output-data", "/opt/ml/processing/output"
    ]
  }
}

# Feature Store
resource "aws_sagemaker_feature_group" "retail_features" {
  feature_group_name             = "retail-customer-features"
  record_identifier_feature_name = "CustomerID"
  event_time_feature_name        = "InvoiceDate"
  role_arn                      = aws_iam_role.sagemaker_role.arn

  feature_definition {
    feature_name = "TotalSpent"
    feature_type = "Fractional"
  }

  feature_definition {
    feature_name = "Frequency"
    feature_type = "Integral"
  }

  feature_definition {
    feature_name = "AvgTransactionValue"
    feature_type = "Fractional"
  }

  feature_definition {
    feature_name = "CustomerLifespan"
    feature_type = "Integral"
  }

  feature_definition {
    feature_name = "AvgPurchaseFrequency"
    feature_type = "Fractional"
  }

  offline_store_config {
    s3_storage_config {
      s3_uri = "s3://${aws_s3_bucket.retail_data_lake.bucket}/feature-store/"
    }
  }

  online_store_config {
    enable_online_store = true
  }
}

# SageMaker Model Execution Role
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "sagemaker-model-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

# SageMaker Model Execution Policy
resource "aws_iam_role_policy" "sagemaker_execution_policy" {
  name = "sagemaker-model-execution-policy"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "sagemaker:CreateModel",
          "sagemaker:CreateEndpoint",
          "sagemaker:CreateEndpointConfig",
          "sagemaker:InvokeEndpoint"
        ]
        Resource = "*"
      }
    ]
  })
}

# Update the SageMaker Model resource
resource "aws_sagemaker_model" "retail_model" {
  name               = "retail-clustering-model"
  execution_role_arn = aws_iam_role.sagemaker_execution_role.arn  # Added this line
  //role_arn           = aws_iam_role.sagemaker_role.arn

  primary_container {
    image = "${aws_ecr_repository.retail_models.repository_url}:latest"
    mode  = "SingleModel"
    model_data_url = "s3://${aws_s3_bucket.retail_data_lake.bucket}/models/clustering-model.tar.gz"
  }

  tags = {
    Environment = "Production"
    Project     = "RetailAnalysis"
  }
}

# SageMaker Endpoint Configuration
resource "aws_sagemaker_endpoint_configuration" "retail_endpoint" {
  name = "retail-clustering-endpoint-config"

  production_variants {
    variant_name           = "AllTraffic"
    model_name            = aws_sagemaker_model.retail_model.name
    instance_type         = "ml.t2.medium"
    initial_instance_count = 1
  }

  tags = {
    Environment = "Production"
    Project     = "RetailAnalysis"
  }
}

# SageMaker Endpoint
resource "aws_sagemaker_endpoint" "retail_endpoint" {
  name                 = "retail-clustering-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.retail_endpoint.name

  tags = {
    Environment = "Production"
    Project     = "RetailAnalysis"
  }
}

# SageMaker Notebook Instance (unchanged)
resource "aws_sagemaker_notebook_instance" "retail_analysis" {
  name                  = "retail-analysis-notebook-demo"
  role_arn              = aws_iam_role.sagemaker_role.arn
  instance_type         = "ml.t3.medium"
  lifecycle_config_name = aws_sagemaker_notebook_instance_lifecycle_configuration.init.name

  tags = {
    Environment = "Production"
    Project     = "RetailAnalysis"
  }
}

# Amazon Managed Grafana Workspace
resource "aws_grafana_workspace" "retail_dashboard" {
  name                     = "retail-analysis-dashboard"
  account_access_type      = "CURRENT_ACCOUNT"
  authentication_providers = ["AWS_SSO"]
  permission_type         = "SERVICE_MANAGED"
  data_sources            = ["CLOUDWATCH", "AMAZON_OPENSEARCH_SERVICE"]
  
  tags = {
    Environment = "Production"
    Project     = "RetailAnalysis"
  }
}

# IAM Role for Grafana
resource "aws_iam_role" "grafana_role" {
  name = "grafana-retail-analysis-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "grafana.amazonaws.com"
        }
      }
    ]
  })
}

# Enhanced SageMaker Role Policy
resource "aws_iam_role_policy" "sagemaker_enhanced_policy" {
  name = "sagemaker-retail-enhanced-policy"
  role = aws_iam_role.sagemaker_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:*",
          "sagemaker:*",
          "ecr:*",
          "cloudwatch:*",
          "logs:*",
          "iam:PassRole"
        ]
        Resource = "*"
      }
    ]
  })
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "retail_dashboard" {
  dashboard_name = "retail-analysis-metrics"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          metrics = [
            ["AWS/SageMaker", "CPUUtilization", "NotebookInstanceName", aws_sagemaker_notebook_instance.retail_analysis.name]
          ]
          period = 300
          stat   = "Average"
          region = "eu-west-2"
          title  = "Notebook CPU Utilization"
        }
      }
    ]
  })
}

# ECR Repository (kept for model storage)
resource "aws_ecr_repository" "retail_models" {
  name = "retail-analysis-demo-models"
}