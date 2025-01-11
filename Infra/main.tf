# main.tf
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

# S3 Bucket for Raw Data Storage
resource "aws_s3_bucket" "retail_data_lake" {
  bucket = "retail-analysis-data-lake"
  
  tags = {
    Environment = "Production"
    Project     = "RetailAnalysis"
  }
}

# Enable versioning for the S3 bucket
resource "aws_s3_bucket_versioning" "retail_data_versioning" {
  bucket = aws_s3_bucket.retail_data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

# SageMaker Notebook Instance with Lifecycle Configuration
resource "aws_sagemaker_notebook_instance_lifecycle_configuration" "init" {
  name = "retail-analysis-lifecycle"
  on_start = base64encode(<<-SCRIPT
    #!/bin/bash
    set -e
    
    # Install requirements
    sudo -u ec2-user -i <<'CONDA_COMMANDS'
    conda activate python3
    pip install pandas>=1.3.0 numpy>=1.21.0 scikit-learn>=1.0.0 matplotlib>=3.4.0 seaborn>=0.11.0 plotly>=5.1.0
    CONDA_COMMANDS
    SCRIPT
  )
}

# SageMaker Notebook Instance
resource "aws_sagemaker_notebook_instance" "retail_analysis" {
  name                    = "retail-analysis-notebook"
  role_arn               = aws_iam_role.sagemaker_role.arn
  instance_type          = "ml.t3.medium"
  lifecycle_config_name  = aws_sagemaker_notebook_instance_lifecycle_configuration.init.name
  
  tags = {
    Environment = "Production"
    Project     = "RetailAnalysis"
  }
}

# IAM Role for SageMaker
resource "aws_iam_role" "sagemaker_role" {
  name = "sagemaker-retail-analysis-role"

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

# IAM Policy for SageMaker
resource "aws_iam_role_policy" "sagemaker_policy" {
  name = "sagemaker-retail-analysis-policy"
  role = aws_iam_role.sagemaker_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.retail_data_lake.arn,
          "${aws_s3_bucket.retail_data_lake.arn}/*"
        ]
      }
    ]
  })
}

# Glue Catalog Database
resource "aws_glue_catalog_database" "retail_db" {
  name = "retail_analysis_db"
}

# Glue Crawler
resource "aws_glue_crawler" "retail_crawler" {
  database_name = aws_glue_catalog_database.retail_db.name
  name          = "retail-data-crawler"
  role          = aws_iam_role.glue_role.arn

  s3_target {
    path = "s3://${aws_s3_bucket.retail_data_lake.bucket}/raw-data/"
  }

  schedule = "cron(0 0 * * ? *)"  # Run daily at midnight
}

# IAM Role for Glue
resource "aws_iam_role" "glue_role" {
  name = "glue-retail-analysis-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "glue.amazonaws.com"
        }
      }
    ]
  })
}

# Athena Workgroup
resource "aws_athena_workgroup" "retail_analysis" {
  name = "retail-analysis-workgroup"

  configuration {
    result_configuration {
      output_location = "s3://${aws_s3_bucket.retail_data_lake.bucket}/athena-results/"
    }
  }
}

# QuickSight User (if needed)
resource "aws_quicksight_user" "analyst" {
  user_name     = "retail-analyst"
  email         = "analyst@yourdomain.com"
  identity_type = "IAM"
  user_role     = "AUTHOR"
  aws_account_id = data.aws_caller_identity.current.account_id
}

# ECR Repository for custom ML models
resource "aws_ecr_repository" "retail_models" {
  name = "retail-analysis-models"
}

# CloudWatch Dashboard for monitoring
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
          region = "us-east-1"
          title  = "Notebook CPU Utilization"
        }
      }
    ]
  })
}