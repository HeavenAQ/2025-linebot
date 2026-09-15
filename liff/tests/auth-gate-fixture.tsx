import React from 'react'
import assert from 'node:assert/strict'
import { renderToStaticMarkup } from 'react-dom/server'
import ExperimentGate from '../src/components/ExperimentGate'

const html = renderToStaticMarkup(
  <ExperimentGate authenticated={false} loginError="LINE 登入驗證未通過"
    onReauthenticate={() => undefined}>
    <div>Private portfolio</div>
  </ExperimentGate>
)
assert.ok(html.includes('重新登入 LINE'))
assert.ok(html.includes('不需要重新登記'))
assert.ok(!html.includes('01 王小明'))
assert.ok(!html.includes('Private portfolio'))
assert.ok(!html.includes('disabled=""'))
console.log('PASS rejected login has an enabled recovery action, no registration prompt, and no private content')
