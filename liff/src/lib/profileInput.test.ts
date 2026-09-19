import assert from 'node:assert/strict'
import test from 'node:test'

import { isValidProfile, normalizeName, normalizeNumber, validateProfile } from './profileInput.ts'

test('accepts what the server accepts', () => {
  for (const [experimentNumber, realName] of [
    ['01', '王小明'],
    ['EG01', '王小明'],
    ['001234', "O'Brien"],
    ['ABCD9', '王 小明']
  ]) {
    assert.ok(isValidProfile({ experimentNumber, realName }), `${experimentNumber} ${realName}`)
  }
})

test('rejects numbers the pattern does not allow', () => {
  for (const experimentNumber of ['', 'abc', 'EG', '1234567', 'EGXYZ1', '01-2']) {
    assert.ok(validateProfile({ experimentNumber, realName: '王小明' }).experimentNumber, experimentNumber)
  }
})

test('rejects names with digits, emoji or too few letters', () => {
  for (const realName of ['', '王', '王小明1', '王小明 😀', '...', '1號']) {
    assert.ok(validateProfile({ experimentNumber: '01', realName }).realName, realName)
  }
})

test('normalizes full-width input and spacing like the server', () => {
  assert.equal(normalizeNumber(' ｅｇ０１ '), 'EG01')
  assert.equal(normalizeName('  王   小明 '), '王 小明')
  assert.ok(isValidProfile({ experimentNumber: 'ｅｇ０１', realName: ' 王小明 ' }))
})
