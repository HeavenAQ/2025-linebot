import assert from 'node:assert/strict'
import test from 'node:test'

import { WorkSchema } from './userData.schema.ts'

const work = {
  date: '2026-08-02-02-15',
  thumbnail: 'https://example.test/thumbnail.jpeg',
  reflection: '',
  preview_note: '',
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
