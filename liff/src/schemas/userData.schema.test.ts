import assert from 'node:assert/strict'
import test from 'node:test'

import { ExpertMatchSchema, WorkSchema } from './userData.schema.ts'

const work = {
  date: '2026-08-02-02-15',
  thumbnail: 'https://example.test/thumbnail.jpeg',
  reflection: '',
  ai_note: '',
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
