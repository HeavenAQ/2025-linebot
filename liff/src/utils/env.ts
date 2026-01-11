export function getRequiredEnv(name: string): string {
  const value = process.env[name]
  if (!value || value.trim() === '') {
    throw new Error(`Environment variable ${name} is not defined`)
  }
  return value
}

export function getBackendBaseUrl(): string {
  const base = getRequiredEnv('BACKEND_BASE_URL')
  return base.replace(/\/+$/, '')
}

