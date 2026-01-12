export function getBackendBaseUrl(): string {
  // Prefer public var for client-side usage; fallback to server var in SSR/dev
  const base =
    (process.env.NEXT_PUBLIC_BACKEND_BASE_URL && process.env.NEXT_PUBLIC_BACKEND_BASE_URL.trim())

  if (!base) {
    throw new Error('Environment variable NEXT_PUBLIC_BACKEND_BASE_URL or BACKEND_BASE_URL is not defined')
  }
  return base.replace(/\/+$/, '')
}
