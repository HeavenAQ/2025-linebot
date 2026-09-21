# Infrastructure (no-LLM variant)

This variant's own GCP resources. The LLM product has its own `infra/` on
`main`; the two share a project, a service account and one GPU, but never each
other's learner data — and keeping the configurations apart is what makes that
true on paper as well as in practice.

## What this branch owns

| Declared here | Declared on `main` |
| --- | --- |
| The bot service `nstc-linebot-2025-noai` | The service account and its roles |
| The queue `analysis-no-llm` | The GPU analysis service and its class-time warm-up |
| `analysis-no-llm-outbox`, `class-stats-no-llm-rebuild` | Artifact Registry and the engine-builder job |
| The bucket `nstc-2025-storage-noai` (uploads and thumbnails only) | The shared secrets (`analysis-grpc-api-key`, `openai-api-key`) |
| The Firestore database `nstc-linebot-noai` | The log-based metrics (project-wide, labelled by service) |
| The secret `2025-linebot-noai-env` | Enabled APIs |

Anything in the right-hand column is read here through a `data` block, never
declared, so the two configurations cannot fight over it.

Rendered analyses do not land in that bucket: the shared GPU service writes to
the LLM product's bucket under this variant's `noai/` storage prefix, which is
what keeps the two deployments' outputs apart.

The GPU service is shared safely because coaching — the only stage that sends
anything to a third party — is switched off per request by this deployment.
Nothing in this configuration turns that on or off; the bot does, through
`ANALYSIS_SKIP_COACHING`.

**Cloud Run templates are ignored after creation.** Terraform creates the
service with a placeholder image and never looks at the template again, so an
`apply` cannot roll production back to whatever image this repository last
named — and with the template go the machine shape and the environment, which
`cd-linebot-no-llm.yml` sets on every revision. What stays Terraform's is what
a deploy does not touch: the service's existence, its region and ingress, who
may invoke it, and deletion protection.

## Applying

State shares the LLM product's bucket under a different prefix, so `main` must
have been applied at least once (it creates the bucket and the service
account):

Terraform needs credentials of its own; the `gcloud` login alone is not enough.
Mint a token from it — an hour's worth, which covers a plan and an apply:

```bash
gcloud auth login
export GOOGLE_OAUTH_ACCESS_TOKEN="$(gcloud auth print-access-token)"

cd infra
terraform init
terraform plan
terraform apply
```

Then add the bot's environment, which Terraform deliberately never sees:

```bash
gcloud secrets versions add 2025-linebot-noai-env --data-file=.env
```

That file points the bot at this variant's own database, bucket and storage
prefix. Getting it wrong is the one mistake that would put this variant's
learners in the other product's data, so check `FIREBASE_DATABASE_ID`,
`GCS_BUCKET_NAME` and the storage prefix before adding a version.

## Adopting the resources that already exist

Everything here predates this configuration, so on the live project each
resource is imported rather than created:

```bash
project=nstc-linebot-2025
region=asia-east1

terraform import google_cloud_run_v2_service.bot "projects/${project}/locations/${region}/services/nstc-linebot-2025-noai"
terraform import google_cloud_tasks_queue.analysis "projects/${project}/locations/${region}/queues/analysis-no-llm"
terraform import google_cloud_scheduler_job.outbox "projects/${project}/locations/${region}/jobs/analysis-no-llm-outbox"
terraform import google_cloud_scheduler_job.class_stats_rebuild "projects/${project}/locations/${region}/jobs/class-stats-no-llm-rebuild"
terraform import google_storage_bucket.learner_media "${project}/nstc-2025-storage-noai"
terraform import google_firestore_database.no_llm "projects/${project}/databases/nstc-linebot-noai"
terraform import google_secret_manager_secret.bot_env "projects/${project}/secrets/2025-linebot-noai-env"
```

Then `terraform plan` and read it. After the imports it proposes two things,
both deliberate: delete protection on the Firestore database, which holds every
score on this branch and was unprotected, and the public invoker binding on the
bot service, which is a re-grant of what is already there — LINE has to reach
the webhook.

## What watches this variant

Alerting is declared on `main` and covers both products: the bot 5xx policy
matches this service by name, and the log-based metrics carry a `service`
label. The spend budget is per project, so it counts this variant's GPU minutes
too. Nothing about alerting is declared here, for the same reason the metrics
are not: two configurations owning one policy is how a policy ends up silently
disabled.

## A plan that wants to destroy something

The bucket and the database hold student recordings and scores.
`prevent_destroy` guards the database; the bucket and the service are guarded
by `deletion_protection`. If a plan proposes replacing any of them, stop and
find out why — it is never the intended change.
