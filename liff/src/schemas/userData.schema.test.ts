import assert from 'node:assert/strict'
import test from 'node:test'

import {
  ExpertMatchSchema,
  PlaybackResponseSchema,
  UserDataSchema,
  WorkSchema
} from './userData.schema.ts'

const work = {
  date: '2026-08-02-02-15',
  thumbnail: 'https://example.test/thumbnail.jpeg',
  reflection: '',
  grading_outcome: {
    grading_details: [],
    total_grade: 90
  }
}

test('normalizes an empty legacy work handedness to right', () => {
  assert.equal(WorkSchema.parse({ ...work, handedness: '' }).handedness, 'right')
})

test('preserves an explicitly left-handed work', () => {
  assert.equal(WorkSchema.parse({ ...work, handedness: 'left' }).handedness, 'left')
})

const expert = {
  expert_id: 'expert-3-1',
  display_name: 'expert-3-1',
  video: {}
}

// Firestore stores nothing for an empty repeated field, so an analysis without
// a warp -- or one recorded before the warp shipped -- arrives with the key
// missing entirely.
test('reads an expert match that carries no alignment', () => {
  assert.deepEqual(ExpertMatchSchema.parse(expert).alignment, [])
  assert.deepEqual(ExpertMatchSchema.parse({ ...expert, alignment: null }).alignment, [])
})

test('keeps a recorded alignment', () => {
  const alignment = [{ normalized_position: 0.5, expert_seconds: 1.25 }]

  assert.deepEqual(ExpertMatchSchema.parse({ ...expert, alignment }).alignment, alignment)
})

const playback = {
  analysis_id: 'analysis-1',
  student_video: {},
  expert,
  timeline: [],
  grade: { grading_details: [], total_grade: 100 }
}

// This deployment's analyses skip the coaching stage, so playback carries
// grades, media and the timeline and nothing else.
test('loads comparison playback without any coaching fields', () => {
  const parsed = PlaybackResponseSchema.parse(playback)
  assert.equal(parsed.analysis_id, 'analysis-1')
  assert.deepEqual(parsed.timeline, [])
})

// What /api/db/user returns for a video still in the analysis queue. The Go
// Firestore client reads the stored empty arrays back as nil slices, which JSON
// encodes as null, and the analysis has written no media or expert yet.
const pendingWork = {
  analysis_status: 'pending',
  analysis_error: '',
  date: '2026-09-14-14-05-31-8f3a2c1d',
  handedness: 'right',
  thumbnail: 'analyses/thumbnail/U1/2026-09-14-14-05-31-8f3a2c1d.jpeg',
  reflection: '尚未填寫心得',
  preview: '',
  grading_outcome: { grading_details: null, total_grade: 0, score_status: 'pending' },
  analysis_id: '8f3a2c1d',
  student_video: {},
  feedback_video: {},
  skeleton_overlay_video: {},
  expert: { expert_id: '', display_name: '', video: {}, timeline: null, alignment: null },
  timeline: null,
  diagnostics: null
}

const user = (serve: Record<string, unknown>) => ({
  portfolio: { serve, smash: null, clear: null, lift: null },
  folder_paths: { root: '', serve: '', smash: '', clear: '', lift: '', thumbnail: '' },
  name: 'Student',
  id: 'U1',
  handedness: 0
})

test('loads the rest of the portfolio while a video is being analyzed', () => {
  const parsed = UserDataSchema.parse(
    user({ '2026-09-01-14-00': { ...work, handedness: 'right' }, [pendingWork.date]: pendingWork })
  )

  assert.deepEqual(Object.keys(parsed.portfolio.serve).sort(), [
    '2026-09-01-14-00',
    pendingWork.date
  ])
  assert.deepEqual(parsed.portfolio.serve[pendingWork.date].grading_outcome.grading_details, [])
})

test('keeps readable attempts when one stored attempt is malformed', () => {
  const parsed = UserDataSchema.parse(
    user({ '2026-09-01-14-00': { ...work, handedness: 'right' }, broken: { date: 42 } })
  )

  assert.deepEqual(Object.keys(parsed.portfolio.serve), ['2026-09-01-14-00'])
})
