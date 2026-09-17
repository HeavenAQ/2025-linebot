/**
 * Writes out/_headers so Netlify serves the security headers with the static
 * export. Runs as `postbuild`; a build that cannot produce the file fails, so a
 * deploy can never go out without its Content-Security-Policy.
 *
 * Plain Node, no dependencies. The policy itself is built by
 * src/lib/securityHeaders.ts, loaded with --experimental-strip-types.
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseEnv } from 'node:util'

import { netlifyHeadersFile } from '../src/lib/securityHeaders.ts'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const outDir = join(root, 'out')
const target = join(outDir, '_headers')

/**
 * The backend URL `next build` compiled in. Next reads .env files for the build
 * but this separate process does not, so resolve it the way Next does: the real
 * environment first, then the production .env files in Next's precedence.
 */
function backendBaseUrl() {
  const name = 'NEXT_PUBLIC_BACKEND_BASE_URL'
  if (process.env[name]?.trim()) return process.env[name]
  for (const file of ['.env.production.local', '.env.local', '.env.production', '.env']) {
    const path = join(root, file)
    if (!existsSync(path)) continue
    const value = parseEnv(readFileSync(path, 'utf8'))[name]
    if (value?.trim()) return value
  }
  return undefined
}

function fail(message) {
  console.error(`write-headers: ${message}`)
  process.exit(1)
}

if (!existsSync(outDir)) fail(`${outDir} does not exist; run next build first`)

let contents
try {
  contents = netlifyHeadersFile(backendBaseUrl())
} catch (error) {
  fail(error instanceof Error ? error.message : String(error))
}

writeFileSync(target, contents)

// Confirm what Netlify will actually read, not what was meant to be written.
const written = existsSync(target) ? readFileSync(target, 'utf8') : ''
if (written !== contents || !written.includes('Content-Security-Policy:')) {
  fail(`${target} was not written`)
}
console.log(`write-headers: wrote ${target}`)
