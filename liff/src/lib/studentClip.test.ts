import assert from 'node:assert/strict'
import test from 'node:test'

import { hasAnalysisRender, playsAnalysisRender } from './studentClip.ts'

test('plays whichever render the student picked', () => {
  const base = { loopActive: false, hasAnalysisRender: true }
  assert.equal(playsAnalysisRender({ ...base, source: 'coach' }), false)
  assert.equal(playsAnalysisRender({ ...base, source: 'analysis' }), true)
})

test('replaying one checkpoint always uses the pause-free render', () => {
  assert.equal(
    playsAnalysisRender({ source: 'coach', loopActive: true, hasAnalysisRender: true }),
    true
  )
})

test('falls back to the coach version when an attempt has no plain render', () => {
  for (const source of ['coach', 'analysis'] as const) {
    for (const loopActive of [false, true]) {
      assert.equal(playsAnalysisRender({ source, loopActive, hasAnalysisRender: false }), false)
    }
  }
})

test('an attempt has a plain render only once its url is signed', () => {
  assert.equal(hasAnalysisRender({ skeleton_overlay_video: { signed_url: 'https://x/a.mp4' } }), true)
  assert.equal(hasAnalysisRender({ skeleton_overlay_video: { signed_url: '' } }), false)
  assert.equal(hasAnalysisRender({ skeleton_overlay_video: undefined }), false)
})
