import test from 'node:test'
import assert from 'node:assert/strict'
import { recoverLiffLogin } from './liffRecovery.ts'

test('LINE ordinary in-app browser clears rejected login and preserves the page', async () => {
  const calls: string[] = []
  const url = 'https://liff-nstc-2025-noai.netlify.app/personal?tab=review&section=reflection'
  await recoverLiffLogin(
    {
      isInClient: () => false,
      isLoggedIn: () => true,
      logout: () => {
        calls.push('logout')
      },
      login: options => {
        calls.push(options.redirectUri)
      },
      permanentLink: {
        createUrlBy: async () => {
          throw new Error('unexpected')
        }
      }
    },
    url,
    () => {
      throw new Error('unexpected')
    }
  )
  assert.deepEqual(calls, ['logout', url])
})

test('LIFF browser uses an official launch URL, never unsupported login/logout', async () => {
  const calls: string[] = []
  const official = 'https://liff.line.me/2006698730-VbnY1UL3/personal?tab=review'
  await recoverLiffLogin(
    {
      isInClient: () => true,
      isLoggedIn: () => true,
      logout: () => {
        throw new Error('unexpected')
      },
      login: () => {
        throw new Error('unexpected')
      },
      permanentLink: {
        createUrlBy: async url => {
          calls.push(url)
          return official
        }
      }
    },
    'https://liff-nstc-2025-noai.netlify.app/personal?tab=review',
    url => calls.push(url)
  )
  assert.equal(calls[1], official)
})
