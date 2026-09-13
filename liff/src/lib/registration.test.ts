import assert from 'node:assert/strict'
import test from 'node:test'
import { hasExperimentRegistration } from './registration.ts'

test('requires persisted registration, not just a successful user fetch', () => {
  const user = { name: 'LINE name', id: 'U1', real_name: '王小明', experiment_number: '001',
    registration_version: 1, registration_completed_at: '2026-09-13T01:02:03Z' }
  assert.equal(hasExperimentRegistration(user), true)
  for (const invalid of [null, {}, { name: 'LINE name' }, { ...user, registration_version: 0 },
    { ...user, real_name: '' }, { ...user, experiment_number: '' },
    { ...user, registration_completed_at: '0001-01-01T00:00:00Z' },
    { ...user, registration_completed_at: 'invalid' }]) {
    assert.equal(hasExperimentRegistration(invalid), false)
  }
})
