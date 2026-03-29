variable "project" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for secret replication"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment (production, staging)"
  type        = string
}

variable "aria_service_account" {
  description = "GCP service account email granted secretAccessor on the ARC API key secret"
  type        = string
}
