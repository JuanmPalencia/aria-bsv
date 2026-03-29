terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

locals {
  instance_name = "aria-${var.environment}"
}

resource "random_password" "db_password" {
  length           = 32
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "google_sql_database_instance" "this" {
  name             = local.instance_name
  project          = var.project
  region           = var.region
  database_version = "POSTGRES_15"

  deletion_protection = var.environment == "production"

  settings {
    tier = var.tier

    ip_configuration {
      ipv4_enabled    = false
      private_network = var.vpc_network
    }

    backup_configuration {
      enabled    = true
      start_time = "03:00"
    }

    insights_config {
      query_insights_enabled = true
    }
  }
}

resource "google_sql_database" "aria" {
  name     = "aria"
  instance = google_sql_database_instance.this.name
  project  = var.project
}

resource "google_sql_user" "aria_user" {
  name     = "aria_user"
  instance = google_sql_database_instance.this.name
  project  = var.project
  password = random_password.db_password.result
}

resource "google_secret_manager_secret" "db_url" {
  secret_id = "${local.instance_name}-database-url"
  project   = var.project

  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }
}

resource "google_secret_manager_secret_version" "db_url" {
  secret = google_secret_manager_secret.db_url.id
  secret_data = "postgresql://${google_sql_user.aria_user.name}:${random_password.db_password.result}@${google_sql_database_instance.this.private_ip_address}:5432/${google_sql_database.aria.name}"
}
