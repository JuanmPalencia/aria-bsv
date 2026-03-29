output "endpoint" {
  description = "RDS endpoint address"
  value       = aws_db_instance.this.endpoint
}

output "port" {
  description = "RDS port"
  value       = aws_db_instance.this.port
}

output "database_name" {
  description = "Database name"
  value       = aws_db_instance.this.db_name
}

output "security_group_id" {
  description = "Security group ID of the RDS instance"
  value       = aws_security_group.rds.id
}

output "ssm_database_url_arn" {
  description = "ARN of the SSM parameter storing the database URL"
  value       = aws_ssm_parameter.db_url.arn
}
