output "instance_connection_name" {
  description = "Cloud SQL instance connection name used by the Cloud SQL Auth Proxy (project:region:instance)"
  value       = google_sql_database_instance.this.connection_name
}

output "database_name" {
  description = "Name of the PostgreSQL database"
  value       = google_sql_database.aria.name
}

output "secret_id" {
  description = "GCP Secret Manager secret ID storing the database connection URL"
  value       = google_secret_manager_secret.db_url.secret_id
}
