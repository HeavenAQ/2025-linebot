/**
 * Which screen the registration gate shows.
 *
 * LIFF login and the backend's registration check each take a moment. Until
 * both have answered, the learner sees a neutral loading screen: showing the
 * registration instructions while the check is still running told registered
 * students to register again, and they did.
 */
export type GateView = 'loading' | 'login-error' | 'error' | 'registration' | 'ready'

export type RegistrationStatus = 'checking' | 'required' | 'ready' | 'error'

export function gateView({
  authenticated,
  loginError,
  status,
  recovering = false
}: {
  authenticated: boolean
  loginError: string | null
  status: RegistrationStatus
  /** A LINE re-login is in progress; the page is about to navigate away. */
  recovering?: boolean
}): GateView {
  if (recovering) return 'loading'
  if (loginError) return 'login-error'
  if (!authenticated) return 'loading'
  switch (status) {
    case 'ready':
      return 'ready'
    case 'required':
      return 'registration'
    case 'error':
      return 'error'
    default:
      return 'loading'
  }
}

/** How long loading may take before the page offers a way out. */
export const SLOW_LOADING_MS = 12_000
