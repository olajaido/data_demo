terraform {
  backend "s3" {
    bucket = "aws-game-demo-terraform-state"
    key    = "terraform/dev/terraform.tfstate"
    region = "eu-west-2"
  }
}