/**
 * The response headers Netlify serves with every page of the static export.
 *
 * A static export has no server to set headers, so scripts/write-headers.mjs
 * writes these into out/_headers after `next build`. The construction lives
 * here, as a pure function, so the policy can be tested without a build.
 *
 * Written with erasable TypeScript only: the build script imports this file
 * straight into plain Node with --experimental-strip-types.
 */

/** LINE's CDN hosts serving the LIFF SDK and the scripts it loads at runtime. */
const LIFF_SCRIPT_HOSTS = ['https://static.line-scdn.net', 'https://liffsdk.line-scdn.net']

/**
 * The origin of the backend the app calls, and nothing else of its URL.
 *
 * CSP source expressions with a path match only that path, and a path in the
 * served headers would publish more of the deployment than the policy needs.
 */
export function backendOrigin(backendBaseUrl: string | undefined): string {
  const raw = backendBaseUrl?.trim()
  if (!raw) {
    throw new Error('NEXT_PUBLIC_BACKEND_BASE_URL is required to build the Content-Security-Policy')
  }
  let url: URL
  try {
    url = new URL(raw)
  } catch {
    throw new Error(`NEXT_PUBLIC_BACKEND_BASE_URL is not an absolute URL: ${raw}`)
  }
  if (url.protocol !== 'https:' && url.protocol !== 'http:') {
    throw new Error(`NEXT_PUBLIC_BACKEND_BASE_URL must be http(s): ${raw}`)
  }
  return url.origin
}

export function contentSecurityPolicy(backendBaseUrl: string | undefined): string {
  const backend = backendOrigin(backendBaseUrl)
  const directives: [string, ...string[]][] = [
    ['default-src', "'self'"],
    // 'unsafe-inline' is unavoidable here: `output: 'export'` inlines the
    // React Server Components bootstrap (self.__next_f.push(...)) as <script>
    // tags in every HTML file, and a static host cannot mint a per-request
    // nonce for them. Hashes would have to be regenerated for every page on
    // every build.
    ['script-src', "'self'", "'unsafe-inline'", ...LIFF_SCRIPT_HOSTS],
    ['style-src', "'self'", "'unsafe-inline'", 'https://static.line-scdn.net'],
    // Profile pictures come from profile.line-scdn.net.
    ['img-src', "'self'", 'data:', 'blob:', 'https://*.line-scdn.net'],
    ['font-src', "'self'", 'data:'],
    // Analysis videos are V4-signed URLs on storage.googleapis.com.
    ['media-src', "'self'", 'blob:', 'https://storage.googleapis.com'],
    // The LIFF SDK talks to api.line.me, access.line.me, liff.line.me and
    // reports usage to uts-front.line-apps.com.
    [
      'connect-src',
      "'self'",
      backend,
      'https://*.line.me',
      'https://*.line-scdn.net',
      'https://uts-front.line-apps.com'
    ],
    // LIFF's subwindow (liff-subwindow.line.me) and login run in frames/forms.
    ['frame-src', 'https://*.line.me'],
    ['form-action', "'self'", 'https://*.line.me'],
    ['base-uri', "'self'"],
    ['object-src', "'none'"],
    ['upgrade-insecure-requests']
    // No frame-ancestors (and no X-Frame-Options below): LINE clients may show
    // the LIFF view inside their own frame, and refusing to be framed would
    // blank the app there.
  ]
  return directives.map(parts => parts.join(' ')).join('; ')
}

/** Every header served for every path, in the order they are written. */
export function securityHeaders(backendBaseUrl: string | undefined): [string, string][] {
  return [
    ['Content-Security-Policy', contentSecurityPolicy(backendBaseUrl)],
    ['Strict-Transport-Security', 'max-age=31536000; includeSubDomains'],
    ['X-Content-Type-Options', 'nosniff'],
    ['Referrer-Policy', 'strict-origin-when-cross-origin'],
    ['Permissions-Policy', 'camera=(), microphone=(), geolocation=(), payment=(), usb=()']
  ]
}

/** The contents of Netlify's _headers file applying the headers to all paths. */
export function netlifyHeadersFile(backendBaseUrl: string | undefined): string {
  const lines = securityHeaders(backendBaseUrl).map(([name, value]) => `  ${name}: ${value}`)
  return ['/*', ...lines, ''].join('\n')
}
