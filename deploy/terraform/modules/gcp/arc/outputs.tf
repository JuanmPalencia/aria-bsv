output "secret_id" {
  description = "GCP Secret Manager secret ID for the TAAL ARC API key"
  value       = google_secret_manager_secret.arc_api_key.secret_id
}

output "secret_name" {
  description = "Full resource name of the Secret Manager secret"
  value       = google_secret_manager_secret.arc_api_key.name
}
