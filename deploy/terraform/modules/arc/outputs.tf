output "ssm_parameter_arn" {
  description = "ARN of the SSM parameter storing the ARC API key"
  value       = aws_ssm_parameter.arc_api_key.arn
}

output "iam_policy_arn" {
  description = "ARN of the IAM policy that allows reading ARC credentials"
  value       = aws_iam_policy.arc_secrets_reader.arn
}
