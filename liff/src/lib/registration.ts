/** Fail closed when a browser is paired with an older, un-gated backend. */
export function hasExperimentRegistration(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false
  const user = value as Record<string, unknown>
  return user.registration_version === 1
    && typeof user.real_name === 'string' && user.real_name.trim().length >= 2
    && typeof user.experiment_number === 'string' && user.experiment_number.trim().length > 0
    && typeof user.registration_completed_at === 'string'
    && Number.isFinite(Date.parse(user.registration_completed_at))
    && Date.parse(user.registration_completed_at) > 0
}
