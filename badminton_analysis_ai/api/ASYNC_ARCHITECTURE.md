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

### Videos sent in coach chat (LLM variant)

A video sent while talking to the coach is the same job with `source: "chat"`,
and three things differ:

- It runs with `skip_coaching`, so the GPU work ends after the skeleton overlay
  and the learner is answered sooner.
- Nothing is replied when the video arrives. The reply token is held for the
  finished answer -- replies are free, pushes are metered -- and the typing
  indicator covers the wait. A token that expires first (LINE honours one for
  about a minute) leaves the answer on the session, delivered with the learner's
  next message.
- A second job, `source: "chat-coaching"`, then runs the coaching pass against
  the same stored input and merges only `coaching_cues`, `ai_note` and
  `feedback_video` into the attempt already recorded. It never rewrites a grade
  or the class aggregate, and it is a queued job rather than a goroutine so an
  instance scaling down cannot take it with it.

Retries use leases and deterministic task names. Completed jobs are no-ops.
Transient inference failures retry up to five actual attempts; invalid motion
is terminal. The one-minute outbox job recovers crashes before task publication
and expires unpublished jobs after 24 hours. It does not poll GPU progress.

## GPU scheduling

`badminton_analysis_ai/api/pose_batcher.py` aggregates up to the TensorRT engine's fixed **16 frames**,
including frames from different videos. A full batch runs immediately; a partial
batch dispatches after at most **50 ms of formation time**. A busy GPU, Cloud
Tasks backlog, network transfer and cold-start time are not covered by 50 ms.
Each request owns its pose getter/tracking state. TensorRT and seeded diffusion
share an execution lock; CPU rendering and LLM calls no longer hold that lock.
Diffusion itself is NOT cross-video batched, preserving its calibrated sampling
contract. Cloud Run admits four concurrent requests to one L4. Each variant's
queue allows two in-flight tasks. Adaptive queue-based tuning is not enabled.

## Configuration and operation

- `ANALYSIS_TASKS_QUEUE`: full queue resource. There is no synchronous
  fallback: without a queue, uploads are refused with a "try again later" reply.
- `ANALYSIS_ASYNC_ACCEPT`: accepts new learner uploads; defaults false. Setting
  it false pauses uploads (learners are told to retry later) while workers keep
  draining already-queued jobs.
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
- Pause new uploads by setting `ANALYSIS_ASYNC_ACCEPT=false`; keep queue
  configuration until all accepted tasks have drained.
- Completing a job also updates the per-skill, per-day class chart aggregate
  (`<data>_class_stats`) in the same transaction; `class-stats-<variant>-rebuild`
  recomputes them nightly at 03:30.
- Existing model weights, 8 candidates, seed 19, EIMD-v3 contract, projection,
  grading and coaching prompts are unchanged. Validate serial/concurrent grades
  and both variant task paths before enabling queue config in production.

Paths are relative to the repository root. Tests:
`badminton_analysis_ai/tests/test_pose_batcher.py`, Go `linebot/api/analysisqueue`,
`linebot/api/db/analysis_jobs_test.go`, `linebot/app/async_analysis_test.go`.
Live Firestore tests opt in and use isolated records.
