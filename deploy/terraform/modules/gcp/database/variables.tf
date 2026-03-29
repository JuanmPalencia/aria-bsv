variable "project" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region where the Cloud SQL instance will be created"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment (production, staging)"
  type        = string
}

variable "tier" {
  description = "Cloud SQL machine tier (e.g. db-g1-small, db-n1-standard-1)"
  type        = string
  default     = "db-g1-small"
}

variable "vpc_network" {
  description = "Self-link of the VPC network for private IP connectivity"
  type        = string
}
