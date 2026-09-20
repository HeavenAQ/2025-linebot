# Infrastructure (LLM product)

Everything this product needs in GCP, described once. The no-LLM variant has
its own `infra/` on `variant/no-llm`: the two share a project and a GPU but
never each other's data, and keeping their configurations apart is what makes
that true on paper as well as in practice.

## What lives here, and what does not

| Managed here | Not managed here |
| --- | --- |
| The service account and its roles | Secret **values** (added with `gcloud`; a value in Terraform is a value in state) |
| The learner bucket, the default Firestore database | Collections and documents — the bot owns those |
| Cloud Run services: the bot, the GPU analysis service, the validation site | Their images, environment and machine shape — GitHub Actions deploys revisions |
| The analysis queue and every scheduled job | LINE channels, the LIFF apps, Netlify |
| Artifact Registry for the TensorRT engine | The engine build recipe (`badminton_analysis_ai/cloudbuild-engine-bootstrap.yaml`) |
| Log-based metrics | Dashboards and alert policies |

**Cloud Run templates are ignored after creation.** Terraform creates each
service with a placeholder image and never looks at the template again, so an
`apply` cannot roll production back to whatever image this repository last
named — and with the template go the machine shape and the environment, which
the deploy workflow sets on every revision. What stays Terraform's is what a
deploy does not touch: the service's existence, its region and ingress, who may
invoke it, and deletion protection. The deploy workflows read the invoker
policy back and fail if it has changed, rather than granting it themselves.

**The no-LLM variant's queue, schedulers, bucket, database and bot service are
declared on its own branch.** The GPU service, the service account, Artifact
Registry, the shared secrets and the log-based metrics are declared here,
because one of each serves both.

## First run on a new project

State lives in a bucket, and Terraform cannot create the bucket it stores its
state in, so that one is made by hand:

```bash
project=nstc-linebot-2025
gcloud storage buckets create "gs://${project}-tfstate" \
  --project "$project" --location asia-east1 --uniform-bucket-level-access
gcloud storage buckets update "gs://${project}-tfstate" --versioning

cd infra
terraform init
terraform import google_storage_bucket.terraform_state "${project}/${project}-tfstate"
terraform plan
```

Then add each secret's value, which Terraform deliberately never sees:

```bash
gcloud secrets versions add 2025-linebot-env --data-file=.env
gcloud secrets versions add openai-api-key --data-file=-
gcloud secrets versions add analysis-grpc-api-key --data-file=-
gcloud secrets versions add gpt-validation-access-codes --data-file=codes.json
```

| Secret | Holds |
| --- | --- |
| `2025-linebot-env` | The bot's whole `.env`: LINE channel credentials, Firestore database and collection names, the OpenAI key |
| `openai-api-key` | Read by the GPU service for the coaching pass |
| `analysis-grpc-api-key` | Shared by the bots and the GPU service, so only the bots may call it |
| `gpt-validation-access-codes` | The expert review site's per-reviewer codes |

Finally deploy the real images by pushing to `main`; the workflows in
`.github/workflows/` own every revision from then on.

## Adopting the resources that already exist

This project was built before this configuration existed, so on the live
project every resource is imported rather than created. Import first, plan,
and expect the plan to be empty apart from fields nobody ever set:

```bash
project=nstc-linebot-2025
region=asia-east1
gpu=asia-southeast1

terraform import google_service_account.bot \
  "projects/${project}/serviceAccounts/nstc-linebot-2025@${project}.iam.gserviceaccount.com"
terraform import google_storage_bucket.learner_media "${project}/nstc-2025-storage"
terraform import google_firestore_database.llm "projects/${project}/databases/(default)"
terraform import 'google_cloud_run_v2_service.bot' "projects/${project}/locations/${region}/services/nstc-linebot-2025"
terraform import google_cloud_run_v2_service.analysis "projects/${project}/locations/${gpu}/services/badminton-analysis-ai"
terraform import google_cloud_run_v2_service.gpt_validation "projects/${project}/locations/${region}/services/gpt-validation"
terraform import google_cloud_tasks_queue.analysis "projects/${project}/locations/${region}/queues/analysis-llm"
terraform import google_artifact_registry_repository.model_artifacts "projects/${project}/locations/${gpu}/repositories/model-artifacts"
terraform import google_cloud_scheduler_job.outbox "projects/${project}/locations/${region}/jobs/analysis-llm-outbox"
terraform import google_cloud_scheduler_job.class_stats_rebuild "projects/${project}/locations/${region}/jobs/class-stats-llm-rebuild"
terraform import 'google_cloud_scheduler_job.gpu_capacity["analysis-gpu-monday-start"]' "projects/${project}/locations/${region}/jobs/analysis-gpu-monday-start"
terraform import 'google_cloud_scheduler_job.gpu_capacity["analysis-gpu-monday-stop"]' "projects/${project}/locations/${region}/jobs/analysis-gpu-monday-stop"
terraform import 'google_cloud_scheduler_job.gpu_warmup["analysis-gpu-warmup"]' "projects/${project}/locations/${region}/jobs/analysis-gpu-warmup"
terraform import 'google_cloud_scheduler_job.gpu_warmup["analysis-gpu-class-warmup"]' "projects/${project}/locations/${region}/jobs/analysis-gpu-class-warmup"

for secret in 2025-linebot-env openai-api-key analysis-grpc-api-key gpt-validation-access-codes; do
  terraform import "google_secret_manager_secret.secrets[\"${secret}\"]" "projects/${project}/secrets/${secret}"
done

for metric in analysis_jobs analysis_job_duration analysis_failures learner_api_refusals; do
  terraform import "google_logging_metric.${metric}" "${metric}"
done
for stage in service llm_inference pose; do
  terraform import "google_logging_metric.analysis_stage_latency[\"${stage}\"]" "analysis_latency_${stage}"
done
```

`google_project_service` and the IAM members are safe to let Terraform create:
enabling an enabled API and granting a granted role both do nothing.

After the imports, the first `apply` on the live project still changes four
things, all of them deliberate:

| Change | Why |
| --- | --- |
| Firestore delete protection **on** | The database holds every score, and was unprotected |
| Public access prevention **enforced** on the state bucket | State names every resource in the project |
| Two descriptions rewritten | The service account and the registry said little |

It also creates the `learner_api_refusals` metric, which never existed.

The engine-builder job (`google_cloud_run_v2_job.engine_builder`) has no
counterpart to import — the engine has been built by hand until now. Cloud Run
checks its image exists when the job is created, and that image comes from the
bootstrap build, so until that build has run once:

```bash
terraform apply -exclude=google_cloud_run_v2_job.engine_builder
```

The learner bucket is described as it is: per-object ACLs, no public access
prevention. Turning both on is a safe tightening — nothing is served from it
except through signed URLs — but it changes the bucket holding every
recording, so it is left as a one-line edit to make on purpose.

## The class schedule

Class is Monday 14:00–18:00 Taipei. The GPU scales to zero, and a cold start
costs the first student of the day a minute or more, so:

- **13:45** capacity reserved (minimum instances 1)
- **13:50 and 13:55** warm-up calls, which run a real inference — the models
  load lazily, so an HTTP ping would not wake them
- **every 10 minutes, 14:00–17:50** kept warm through the session
- **18:10** capacity released

Change the times in `scheduler.tf`, not in the console: the console change is
the one nobody remembers next term.

## Routine use

```bash
cd infra
terraform plan      # read it; an unexpected replace is the thing to catch
terraform apply
```

A plan that wants to replace a Cloud Run service, a bucket or a database is
wrong — those carry student data or a public URL. `prevent_destroy` guards the
worst of them; read the rest.
