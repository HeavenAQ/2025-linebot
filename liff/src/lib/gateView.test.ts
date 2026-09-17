import assert from 'node:assert/strict'
import test from 'node:test'

import { gateView } from './gateView.ts'

test('never shows registration instructions before the check has answered', () => {
  assert.equal(gateView({ authenticated: false, loginError: null, status: 'checking' }), 'loading')
  assert.equal(gateView({ authenticated: false, loginError: null, status: 'required' }), 'loading')
  assert.equal(gateView({ authenticated: true, loginError: null, status: 'checking' }), 'loading')
})

test('shows registration only when the backend says the learner is unregistered', () => {
  assert.equal(
    gateView({ authenticated: true, loginError: null, status: 'required' }),
    'registration'
  )
  assert.equal(gateView({ authenticated: true, loginError: null, status: 'ready' }), 'ready')
  assert.equal(gateView({ authenticated: true, loginError: null, status: 'error' }), 'error')
})

test('a login problem takes precedence over everything else', () => {
  assert.equal(
    gateView({ authenticated: false, loginError: 'x', status: 'checking' }),
    'login-error'
  )
  assert.equal(gateView({ authenticated: true, loginError: 'x', status: 'ready' }), 'login-error')
})

test('a LINE re-login in progress shows loading, not the error it is fixing', () => {
  assert.equal(
    gateView({ authenticated: false, loginError: 'x', status: 'checking', recovering: true }),
    'loading'
  )
})
