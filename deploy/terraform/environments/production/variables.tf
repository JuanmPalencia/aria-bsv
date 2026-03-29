variable "aws_region" {
  description = "AWS region for production deployment"
  type        = string
  default     = "us-east-1"
}

variable "vpc_id" {
  description = "VPC ID for production environment"
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs (min 2 AZs)"
  type        = list(string)
}

variable "app_security_group_ids" {
  description = "App tier security group IDs allowed to reach RDS"
  type        = list(string)
}

variable "arc_api_key" {
  description = "TAAL ARC API key"
  type        = string
  sensitive   = true
}
