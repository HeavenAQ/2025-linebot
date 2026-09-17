import assert from 'node:assert/strict'
import test from 'node:test'

import {
  jwtExpiry,
  rateLimitMessage,
  RATE_LIMITED_FALLBACK,
  selectCredential
} from './credentials.ts'

const base64url = (text: string) =>
  Buffer.from(text, 'utf8')
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '')

const jwt = (payload: unknown) =>
  `${base64url('{"alg":"HS256","typ":"JWT"}')}.${base64url(JSON.stringify(payload))}.signature`

const NOW_MS = 1_800_000_000_000
const NOW = NOW_MS / 1000

test('sends a fresh ID token as a bearer token', () => {
  const idToken = jwt({ exp: NOW + 3600, name: '王小明' })
  assert.deepEqual(selectCredential(idToken, 'access', NOW_MS), {
    kind: 'id_token',
    header: 'Authorization',
    value: `Bearer ${idToken}`
  })
})

test('switches to the access token within a minute of the ID token expiring', () => {
  const access = { kind: 'access_token', header: 'X-Line-Access-Token', value: 'access' }
  assert.deepEqual(selectCredential(jwt({ exp: NOW + 60 }), 'access', NOW_MS), access)
  assert.deepEqual(selectCredential(jwt({ exp: NOW + 30 }), 'access', NOW_MS), access)
  assert.deepEqual(selectCredential(jwt({ exp: NOW - 10 }), 'access', NOW_MS), access)
  assert.equal(selectCredential(jwt({ exp: NOW + 61 }), 'access', NOW_MS)?.kind, 'id_token')
})

test('uses the access token when there is no ID token', () => {
  assert.equal(selectCredential(null, 'access', NOW_MS)?.kind, 'access_token')
  assert.equal(selectCredential('', 'access', NOW_MS)?.kind, 'access_token')
})

test('sends nothing when neither token is usable', () => {
  assert.equal(selectCredential(null, null, NOW_MS), null)
  assert.equal(selectCredential(jwt({ exp: NOW - 10 }), null, NOW_MS), null)
  assert.equal(selectCredential(jwt({ exp: NOW + 3600 }), '', NOW_MS)?.kind, 'id_token')
})

test('treats a malformed ID token as expired', () => {
  const malformed = [
    'not-a-jwt',
    'a.b',
    'a..c',
    'a.b.c.d',
    `x.${base64url('not json')}.y`,
    `x.${base64url('null')}.y`,
    `x.${base64url('"string"')}.y`,
    jwt({}),
    jwt({ exp: String(NOW + 3600) }),
    'x.%%%%.y'
  ]
  for (const token of malformed) {
    assert.equal(jwtExpiry(token), null, token)
    assert.equal(selectCredential(token, 'access', NOW_MS)?.kind, 'access_token', token)
    assert.equal(selectCredential(token, null, NOW_MS), null, token)
  }
})

test('reads exp from base64url payloads that need padding and carry UTF-8', () => {
  for (const name of ['a', 'ab', 'abc', '王小明', '??>>']) {
    assert.equal(jwtExpiry(jwt({ name, exp: 1234 })), 1234)
  }
})

test('shows the backend message for a 429, else a fallback', () => {
  assert.equal(
    rateLimitMessage({ error: '上傳太頻繁，請一分鐘後再試。' }, '60'),
    '上傳太頻繁，請一分鐘後再試。'
  )
  assert.equal(rateLimitMessage({ error: '' }, '30'), '操作太頻繁，請於 30 秒後再試。')
  assert.equal(rateLimitMessage(null, null), RATE_LIMITED_FALLBACK)
  assert.equal(rateLimitMessage({}, 'Wed, 21 Oct 2015 07:28:00 GMT'), RATE_LIMITED_FALLBACK)
})
