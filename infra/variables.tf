variable "project_id" {
  description = "GCP project, shared with the LLM product."
  type        = string
  default     = "nstc-linebot-2025"
}

variable "region" {
  description = "Region for this variant's bot, queue, schedulers and learner data."
  type        = string
  default     = "asia-east1"
}

variable "service_account_id" {
  description = <<-EOT
    The account both products run as. It is created by the LLM product's
    configuration on `main` and only read here, so neither configuration can
    take it out from under the other.
  EOT
  type        = string
  default     = "nstc-linebot-2025"
}

variable "class_timezone" {
  description = "Timezone the schedule is written in."
  type        = string
  default     = "Asia/Taipei"
}

variable "placeholder_image" {
  description = <<-EOT
    Image used when Terraform first creates the service. The real image arrives
    from GitHub Actions, and Terraform ignores it afterwards.
  EOT
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}
