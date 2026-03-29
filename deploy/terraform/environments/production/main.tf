terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket         = "aria-bsv-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "aria-bsv-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = local.common_tags
  }
}

locals {
  environment = "production"
  common_tags = {
    Project     = "aria-bsv"
    Environment = local.environment
    ManagedBy   = "terraform"
  }
}

module "arc" {
  source = "../../modules/arc"

  environment = local.environment
  aws_region  = var.aws_region
  arc_api_key = var.arc_api_key
  tags        = local.common_tags
}

module "database" {
  source = "../../modules/database"

  environment                = local.environment
  vpc_id                     = var.vpc_id
  subnet_ids                 = var.private_subnet_ids
  allowed_security_group_ids = var.app_security_group_ids
  instance_class             = "db.t3.small"
  allocated_storage          = 50
  backup_retention_days      = 14
  tags                       = local.common_tags
}
