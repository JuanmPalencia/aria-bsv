variable "environment" {
  description = "Deployment environment (production, staging)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "arc_api_key" {
  description = "TAAL ARC API key"
  type        = string
  sensitive   = true
}

variable "tags" {
  description = "Common resource tags"
  type        = map(string)
  default     = {}
}
