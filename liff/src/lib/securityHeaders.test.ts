import assert from 'node:assert/strict'
import test from 'node:test'

import {
  backendOrigin,
  contentSecurityPolicy,
  netlifyHeadersFile,
  securityHeaders
} from './securityHeaders.ts'

const BACKEND = 'https://nstc-linebot-2025-bfn4xpszya-de.a.run.app'

const directives = (policy: string) =>
  new Map(
    policy.split('; ').map(entry => {
      const [name, ...sources] = entry.split(' ')
      return [name, sources] as const
    })
  )

test('connects only to the backend origin, never its path or query', () => {
  assert.equal(backendOrigin(`${BACKEND}/api/v1/?token=secret#x`), BACKEND)
  const policy = contentSecurityPolicy(`${BACKEND}/api/v1/?token=secret`)
  assert.ok(directives(policy).get('connect-src')?.includes(BACKEND))
  assert.ok(!policy.includes('/api/v1'))
  assert.ok(!policy.includes('secret'))
})

test('refuses to build a policy without a usable backend URL', () => {
  assert.throws(() => contentSecurityPolicy(undefined), /NEXT_PUBLIC_BACKEND_BASE_URL/)
  assert.throws(() => contentSecurityPolicy('   '), /NEXT_PUBLIC_BACKEND_BASE_URL/)
  assert.throws(() => contentSecurityPolicy('nstc.example.com'), /absolute URL/)
  assert.throws(() => contentSecurityPolicy('javascript:alert(1)'), /http/)
})

test('carries every directive the LIFF app depends on', () => {
  const policy = directives(contentSecurityPolicy(BACKEND))
  assert.deepEqual(policy.get('default-src'), ["'self'"])
  assert.deepEqual(policy.get('script-src'), [
    "'self'",
    "'unsafe-inline'",
    'https://static.line-scdn.net',
    'https://liffsdk.line-scdn.net'
  ])
  assert.ok(policy.get('media-src')?.includes('https://storage.googleapis.com'))
  assert.ok(policy.get('img-src')?.includes('https://*.line-scdn.net'))
  // The video poster is a signed Cloud Storage image; without this the player
  // shows an empty frame until the video paints, which the poster exists to avoid.
  assert.ok(policy.get('img-src')?.includes('https://storage.googleapis.com'))
  for (const host of ['https://*.line.me', 'https://uts-front.line-apps.com']) {
    assert.ok(policy.get('connect-src')?.includes(host), host)
  }
  assert.deepEqual(policy.get('frame-src'), ['https://*.line.me'])
  assert.deepEqual(policy.get('object-src'), ["'none'"])
  assert.deepEqual(policy.get('base-uri'), ["'self'"])
  assert.ok(policy.has('upgrade-insecure-requests'))
  // LINE may embed the view; being framed must stay allowed.
  assert.ok(!policy.has('frame-ancestors'))
})

test('serves the hardening headers on every path', () => {
  const headers = new Map(securityHeaders(BACKEND))
  assert.equal(headers.get('Strict-Transport-Security'), 'max-age=31536000; includeSubDomains')
  assert.equal(headers.get('X-Content-Type-Options'), 'nosniff')
  assert.equal(headers.get('Referrer-Policy'), 'strict-origin-when-cross-origin')
  assert.equal(
    headers.get('Permissions-Policy'),
    'camera=(), microphone=(), geolocation=(), payment=(), usb=()'
  )
  assert.ok(!headers.has('X-Frame-Options'))

  const file = netlifyHeadersFile(BACKEND)
  const [path, ...lines] = file.trimEnd().split('\n')
  assert.equal(path, '/*')
  assert.equal(lines.length, 5)
  for (const line of lines) assert.match(line, /^ {2}[A-Za-z-]+: \S/)
})
