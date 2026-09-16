# Queued analysis: LLM and no-LLM

The variants retain their existing public URLs, Firestore databases and storage
prefixes. Only the queue/scheduler terminology changes; no learner data moves.

## Request lifecycle

1. LINE webhook downloads the upload, stores the original under
   `analyses/input/<user>/<job>.mp4` in that variant's own bucket, and uploads a
   unique thumbnail. The thumbnail is never overwritten after LINE sees it.
2. A Firestore transaction writes the pending portfolio entry and job/outbox.
   The job ID hashes the deployment collection, learner ID and LINE message ID.
3. Go publishes a named Cloud Task and replies with the pending portfolio.
   A thumbnail is available now; scores are not presented as zero or final.
4. Cloud Tasks calls Go's authenticated HTTP worker. This is an HTTP-to-gRPC
   bridge, not a polling loop. It downloads the stored source and performs one
   GPU RPC. A transaction updates only analysis fields, preserving notes.
5. LINE's reply token is never reused. The existing card's buttons fetch current
   data; LIFF refreshes pending records every five seconds while visible. The
   original LINE card itself remains a pending receipt (LINE messages cannot
   be edited to replace its cached score text).

Retries use leases and deterministic task names. Completed jobs are no-ops.
Transient inference failures retry up to five actual attempts; invalid motion
is terminal. The one-minute outbox job recovers crashes before task publication
and expires unpublished jobs after 24 hours. It does not poll GPU progress.

## GPU scheduling

`pose_batcher.py` aggregates up to the TensorRT engine's fixed **16 frames**,
including frames from different videos. A full batch runs immediately; a partial
batch dispatches after at most **50 ms of formation time**. A busy GPU, Cloud
Tasks backlog, network transfer and cold-start time are not covered by 50 ms.
Each request owns its pose getter/tracking state. TensorRT and seeded diffusion
share an execution lock; CPU rendering and LLM calls no longer hold that lock.
Diffusion itself is NOT cross-video batched, preserving its calibrated sampling
contract. Cloud Run admits four concurrent requests to one L4. Each variant's
queue allows two in-flight tasks. Adaptive queue-based tuning is not enabled.

## Configuration and operation

- `ANALYSIS_TASKS_QUEUE`: full queue resource; empty retains synchronous fallback.
- `ANALYSIS_ASYNC_ACCEPT`: explicitly enable queued learner uploads after smoke
  tests; defaults false. Workers remain available to drain when this is false.
- `ANALYSIS_WORKER_URL`: canonical Go service URL (OIDC audience).
- `ANALYSIS_TASK_SERVICE_ACCOUNT`: exact accepted service-account identity.
- The internal worker, outbox, warmup and capacity routes verify Google OIDC,
  audience, service-account email and email verification; task headers alone
  grant no authority. Learner routes retain LINE identity/registration checks.
- `scripts/configure_async_analysis.sh` provisions queues and schedules after
  the new services have deployed. Run once from either branch.
- Taiwan (`Asia/Taipei`) Mondays: min GPU instances 1 at **13:45**, model warmup
  at **13:50/13:55** and every ten minutes during class, min instances 0 at
  **18:10**. A warmup executes pose inference, not just a health ping. Reserved
  GPU time is billable. Scale-to-zero remains available outside this window.
- Disable new async acceptance by setting `ANALYSIS_ASYNC_ACCEPT=false`; keep
  queue configuration until all accepted tasks have drained.
- Existing model weights, 8 candidates, seed 19, EIMD-v3 contract, projection,
  grading and coaching prompts are unchanged. Validate serial/concurrent grades
  and both variant task paths before enabling queue config in production.

Tests: `tests/test_pose_batcher.py`, Go `api/analysisqueue`, `api/db/analysis_jobs_test.go`,
`app/async_analysis_test.go`. Live Firestore tests opt in and use isolated records.
