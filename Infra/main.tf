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

# VPC and Networking
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "retail-analysis-vpc"
  }
}

# Public Subnet
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "eu-west-2a"
  map_public_ip_on_launch = true

  tags = {
    Name = "retail-analysis-public"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "retail-analysis-igw"
  }
}

# Route Table
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

# S3 Bucket for Raw Data Storage
resource "aws_s3_bucket" "retail_data_lake" {
  bucket = "retail-analysis-data-demo"

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
  name = "retail-analysis-lifecycle-demo"
  on_start = base64encode(<<-SCRIPT
    #!/bin/bash
    set -e
    
    # Install requirements
    sudo -u ec2-user -i <<'CONDA_COMMANDS'
    conda activate python3
    pip install pandas>=1.3.0 numpy>=1.21.0 scikit-learn>=1.0.0 matplotlib>=3.4.0 seaborn>=0.11.0 plotly>=5.1.0 streamlit>=1.0.0
    CONDA_COMMANDS
    SCRIPT
  )
}

# SageMaker Notebook Instance
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

# IAM Role for SageMaker
resource "aws_iam_role" "sagemaker_role" {
  name = "sagemaker-retail-analysis-demo-role"

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
  name = "sagemaker-retail-analysis-demo-policy"
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
  name = "retail_analysis_demo_b"
}

# Glue Crawler
resource "aws_glue_crawler" "retail_crawler" {
  database_name = aws_glue_catalog_database.retail_db.name
  name          = "retail-data-demo-crawler"
  role          = aws_iam_role.glue_role.arn

  s3_target {
    path = "s3://${aws_s3_bucket.retail_data_lake.bucket}/raw-data/"
  }

  schedule = "cron(0 0 * * ? *)" # Run daily at midnight
}

# IAM Role for Glue
resource "aws_iam_role" "glue_role" {
  name = "glue-retail-analysis-demo-role"

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
  name = "retail-analysis-demo-workgroup"

  configuration {
    result_configuration {
      output_location = "s3://${aws_s3_bucket.retail_data_lake.bucket}/athena-results/"
    }
  }
}

# ECR Repository for Dashboard and ML models
resource "aws_ecr_repository" "retail_models" {
  name = "retail-analysis-demo-models"
}

# ECS Cluster for Dashboard
resource "aws_ecs_cluster" "dashboard" {
  name = "retail-dashboard-demo-cluster"
}

# Security Group for ECS Tasks
resource "aws_security_group" "ecs_tasks" {
  name        = "retail-dashboard-ecs-tasks"
  description = "Allow inbound traffic to Streamlit dashboard"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "Streamlit port"
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "dashboard" {
  family                   = "retail-dashboard"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name  = "dashboard"
      image = "${aws_ecr_repository.retail_models.repository_url}:latest"
      portMappings = [
        {
          containerPort = 8501
          protocol      = "tcp"
        }
      ]
      environment = [
        {
          name  = "AWS_DEFAULT_REGION"
          value = "eu-west-2"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/retail-dashboard"
          "awslogs-region"        = "eu-west-2"
          "awslogs-stream-prefix" = "ecs"
        }
      }
    }
  ])
}

# IAM Role for ECS Task Execution
resource "aws_iam_role" "ecs_execution_role" {
  name = "retail-dashboard-execution-demo-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

# IAM Role for ECS Tasks
resource "aws_iam_role" "ecs_task_role" {
  name = "retail-dashboard-task-demo-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

# ECS Service
resource "aws_ecs_service" "dashboard" {
  name            = "retail-dashboard-demo"
  cluster         = aws_ecs_cluster.dashboard.id
  task_definition = aws_ecs_task_definition.dashboard.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = [aws_subnet.public.id]
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = true
  }
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
          region = "eu-west-2"
          title  = "Notebook CPU Utilization"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          metrics = [
            ["AWS/ECS", "CPUUtilization", "ServiceName", aws_ecs_service.dashboard.name, "ClusterName", aws_ecs_cluster.dashboard.name]
          ]
          period = 300
          stat   = "Average"
          region = "eu-west-2"
          title  = "Dashboard CPU Utilization"
        }
      }
    ]
  })
}

# CloudWatch Log Group for ECS
resource "aws_cloudwatch_log_group" "dashboard" {
  name              = "/ecs/retail-dashboard"
  retention_in_days = 30
}