terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# SSM Parameter for ARC API key
resource "aws_ssm_parameter" "arc_api_key" {
  name        = "/${var.environment}/aria/arc_api_key"
  description = "TAAL ARC API key for BSV transaction broadcasting"
  type        = "SecureString"
  value       = var.arc_api_key

  tags = var.tags
}

# IAM policy to read ARC secrets
resource "aws_iam_policy" "arc_secrets_reader" {
  name        = "aria-arc-secrets-reader-${var.environment}"
  description = "Allows reading ARC credentials from SSM"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters"
        ]
        Resource = aws_ssm_parameter.arc_api_key.arn
      },
      {
        Effect = "Allow"
        Action = "kms:Decrypt"
        Resource = "*"
        Condition = {
          StringEquals = {
            "kms:ViaService" = "ssm.${var.aws_region}.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = var.tags
}
