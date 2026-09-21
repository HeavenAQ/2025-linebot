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
| — it holds **both** products' rendered analyses, since the shared GPU service writes to its own bucket and separates deployments by storage prefix | — |
| Cloud Run services: the bot, the GPU analysis service, the validation site | Their images, environment and machine shape — GitHub Actions deploys revisions |
| The analysis queue and every scheduled job | LINE channels, the LIFF apps, Netlify |
| Artifact Registry, and the recipe that builds the engine image (`cloudbuild/engine-bootstrap.yaml`) | The engine itself — built by running the job, not by an apply |
| Log-based metrics, alert policies, the spend budget | Dashboards |
| How GitHub Actions authenticates (the identity pool and provider) | GitHub repository secrets |

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

## Before anything: how Terraform authenticates

Terraform needs Google credentials of its own; the `gcloud` login alone is not
enough. The simplest way, and the one the applies here were run with, is to
hand it a token minted from that login:

```bash
gcloud auth login                                    # once, as the project owner
export GOOGLE_OAUTH_ACCESS_TOKEN="$(gcloud auth print-access-token)"
```

The token lasts about an hour — long enough for a plan and an apply. Re-export
it when Terraform starts reporting 401s.

**Apply as yourself, not as the deploy service account.** The key at
`linebot/sa-key.json` works for everything except the budget: managing budgets
is a permission on the *billing account*, and only the owner account holds
`roles/billing.admin` there. Applying with the key gets as far as the budget and
then stops. (If you ever do want the key to manage it, grant the narrower
`roles/billing.costsManager` — see "The budget" below.)

## Setting it up, start to finish

The commands below build the whole project from nothing. On the existing
project, skip to "Adopting the resources that already exist" — every one of
these resources is already there and must be imported instead of created.

**1. The state bucket.** Terraform cannot create the bucket it keeps its own
state in, so this one is made by hand:

```bash
project=nstc-linebot-2025
gcloud storage buckets create "gs://${project}-tfstate" \
  --project "$project" --location asia-east1 --uniform-bucket-level-access
gcloud storage buckets update "gs://${project}-tfstate" --versioning
```

**2. The values this repository does not carry.** It is public, so the alert
address and the billing account ID live in a gitignored file:

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars
$EDITOR terraform.tfvars        # alert_email, billing_account_id
```

**3. Create everything.**

```bash
terraform init
terraform import google_storage_bucket.terraform_state "${project}/${project}-tfstate"
terraform plan                  # read it
terraform apply
```

**4. Fill the secrets**, which Terraform declares but never sees (a value in
Terraform is a value in state):

```bash
gcloud secrets versions add 2025-linebot-env --data-file=.env
gcloud secrets versions add openai-api-key --data-file=-
gcloud secrets versions add analysis-grpc-api-key --data-file=-
gcloud secrets versions add gpt-validation-access-codes --data-file=codes.json
```

**5. Deploy the real images** by pushing to `main`. Until then each service runs
the placeholder image Terraform created it with, and the workflows in
`.github/workflows/` own every revision from that point on.

**6. Build the TensorRT engine** once the GPU service exists, then turn its job
on — Cloud Run refuses to create a job whose image does not exist yet, which is
why `create_engine_builder_job` starts false:

```bash
gcloud builds submit --config cloudbuild/engine-bootstrap.yaml \
  --project "$project" ../badminton_analysis_ai
echo 'create_engine_builder_job = true' >> terraform.tfvars
terraform apply
gcloud run jobs execute rfdetr-engine-builder \
  --region asia-southeast1 --project "$project" --wait
```

**7. Point GitHub at the pool.** `wif.tf` creates the identity pool and
provider; the repository needs their full names as secrets:

```bash
num="$(gcloud projects describe "$project" --format='value(projectNumber)')"
gh secret set GCP_WORKLOAD_IDENTITY_PROVIDER --body \
  "projects/${num}/locations/global/workloadIdentityPools/git-actions-pool/providers/git-actions-provider"
gh secret set GCP_SA_EMAIL --body "nstc-linebot-2025@${project}.iam.gserviceaccount.com"
gh secret set GCP_PROJECT_ID --body "$project"
```

| Secret | Holds |
| --- | --- |
| `2025-linebot-env` | The bot's whole `.env`: LINE channel credentials, Firestore database and collection names, the OpenAI key |
| `openai-api-key` | Read by the GPU service for the coaching pass |
| `analysis-grpc-api-key` | Shared by the bots and the GPU service, so only the bots may call it |
| `gpt-validation-access-codes` | The expert review site's per-reviewer codes |

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

num=38977606134
principal="principalSet://iam.googleapis.com/projects/${num}/locations/global/workloadIdentityPools/git-actions-pool/attribute.repository/HeavenAQ/2025-linebot"
sa="projects/${project}/serviceAccounts/nstc-linebot-2025@${project}.iam.gserviceaccount.com"

terraform import google_iam_workload_identity_pool.github \
  "projects/${project}/locations/global/workloadIdentityPools/git-actions-pool"
terraform import google_iam_workload_identity_pool_provider.github \
  "projects/${project}/locations/global/workloadIdentityPools/git-actions-pool/providers/git-actions-provider"
terraform import google_service_account_iam_member.github_impersonates_bot \
  "${sa} roles/iam.workloadIdentityUser ${principal}"
terraform import google_service_account_iam_member.github_mints_tokens_as_bot \
  "${sa} roles/iam.serviceAccountTokenCreator ${principal}"

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

It also creates what never existed at all: the `learner_api_refusals` and
`scheduler_job_failures` metrics, five alert policies, the email notification
channel and the budget.

The engine-builder job (`google_cloud_run_v2_job.engine_builder`) has no
counterpart to import — the engine has been built by hand until now. Cloud Run
checks its image exists when the job is created, and that image comes from the
bootstrap build, so the job is behind a flag that stays off until that build has
run once:

```hcl
create_engine_builder_job = true   # in terraform.tfvars, after the bootstrap build
```

The learner bucket is described as it is: per-object ACLs, no public access
prevention. Turning both on is a safe tightening — nothing is served from it
except through signed URLs — but it changes the bucket holding every
recording, so it is left as a one-line edit to make on purpose.

## The budget

It replaces a `$10 Monthly Budget Alert` that covered the whole billing account
and warned at NT$5, NT$9, NT$10, NT$15 and NT$100 — often enough that the mail
stopped meaning anything. This one is scoped to this project, so Maps Platform
spend on the same account is counted by its own default budgets and not by
this one. Widen `budget_filter.projects` if that ever stops being what you
want.

It is the only resource here billed against a *billing account* rather than the
project, and that makes it the only one with its own requirements:

- **It needs the owner account.** `roles/billing.admin` on the billing account
  is held by one user; the deploy service account has no billing role at all.
  To let the key manage it instead, grant the narrower role once:

  ```bash
  gcloud billing accounts add-iam-policy-binding 019A8A-F51497-B2EE02 \
    --member "serviceAccount:nstc-linebot-2025@nstc-linebot-2025.iam.gserviceaccount.com" \
    --role roles/billing.costsManager
  ```

- **It is applied through an aliased provider** (`google.billing` in
  `versions.tf`) with `user_project_override`. Requests to the budgets API must
  name a project to charge quota to, and user credentials carry none by default;
  without the alias the API replies that the service is disabled on a project
  you have never heard of. Only the budget uses that provider, so nothing else
  needs the extra permission it implies.

## Who may deploy

`wif.tf` is worth reading before anything else here. GitHub Actions holds no key
for this project: it presents its own OIDC token, and the provider's attribute
condition decides whether to believe it. That condition is a single line —

```
assertion.repository=='HeavenAQ/2025-linebot'
```

— and it is the only thing standing between a GitHub account and a service
account that can read every learner's recordings and scores. Narrowing it
further (to a branch, or a deployment environment) is safe; widening it is the
change to think hardest about in this whole directory.

## Alerts

Five conditions, all to one email channel, all with loose thresholds — an alert
that fires on a single transient failure gets muted, and a muted alert is worse
than none:

| Alert | Fires when |
| --- | --- |
| LINE bot serving 5xx | More than 3 server errors in 5 minutes on either bot |
| Analyses failing | More than 3 failures in 10 minutes |
| Scheduled job failing | More than 5 failed attempts of one job in 15 minutes |
| GPU instance held for six hours | Class reserves one 13:45–18:10; six hours means the stop job did not take |
| Learner API refusing in bulk | 50 refusals in 5 minutes — a probe, or a retry loop in the app |

The budget is NT$1,000 a month and mails at every full multiple of it: 1,000,
2,000, and so on to 5,000. It caps nothing — GCP keeps serving when a threshold
trips. The L4 is why it exists: it bills by the second for as long as an
instance is up, so a stop job that silently failed is the difference between a
normal month and an unpleasant one.

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
