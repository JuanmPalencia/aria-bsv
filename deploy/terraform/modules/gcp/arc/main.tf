terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

locals {
  secret_name = "${var.environment}-aria-arc-api-key"
}

resource "google_secret_manager_secret" "arc_api_key" {
  secret_id = local.secret_name
  project   = var.project

  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }
}

resource "google_secret_manager_secret_version" "arc_api_key" {
  secret      = google_secret_manager_secret.arc_api_key.id
  secret_data = "REPLACE_ME"

  lifecycle {
    ignore_changes = [secret_data]
  }
}

resource "google_secret_manager_iam_member" "aria_service_sa" {
  secret_id = google_secret_manager_secret.arc_api_key.secret_id
  project   = var.project
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.aria_service_account}"
}
