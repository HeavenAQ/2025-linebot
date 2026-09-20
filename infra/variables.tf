variable "project_id" {
  description = "GCP project that holds every resource here."
  type        = string
  default     = "nstc-linebot-2025"
}

variable "region" {
  description = "Region for the bot, its queues and schedulers, and learner data."
  type        = string
  default     = "asia-east1"
}

variable "gpu_region" {
  description = <<-EOT
    Region for the analysis service. It is separate because NVIDIA L4 capacity
    on Cloud Run is not offered in asia-east1; the extra hop costs a few
    milliseconds on an RPC that already takes seconds.
  EOT
  type        = string
  default     = "asia-southeast1"
}

variable "service_account_id" {
  description = "Account every service runs as. One identity, scoped by role."
  type        = string
  default     = "nstc-linebot-2025"
}

variable "class_timezone" {
  description = "Timezone the class schedule is written in."
  type        = string
  default     = "Asia/Taipei"
}

variable "accept_uploads" {
  description = <<-EOT
    Whether the bots accept new uploads. Set false to pause intake while
    queued work drains; learners are told to try again later.
  EOT
  type        = bool
  default     = true
}

variable "placeholder_image" {
  description = <<-EOT
    Image used when Terraform first creates a Cloud Run service. The real image
    arrives from GitHub Actions, and Terraform ignores it afterwards, so this
    only ever runs on a brand-new project.
  EOT
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}

variable "github_repository" {
  description = "The repository allowed to deploy, as owner/name. Anything else the identity provider refuses."
  type        = string
  default     = "HeavenAQ/2025-linebot"
}

# Kept out of the repository, which is public: set both in terraform.tfvars,
# which is gitignored. See README.md.
variable "alert_email" {
  description = "Where alert and budget notifications are sent."
  type        = string
}

variable "billing_account_id" {
  description = "Billing account the budget is attached to, as XXXXXX-XXXXXX-XXXXXX."
  type        = string
}
