type RecoverySDK = {
  isInClient: () => boolean
  isLoggedIn: () => boolean
  logout: () => void
  login: (_options: { redirectUri: string }) => void
  permanentLink: { createUrlBy: (_url: string) => Promise<string> }
}

// A Netlify link inside LINE can use the ordinary in-app browser rather than
// the LIFF browser. Recovery must use the SDK's actual browser mode.
export async function recoverLiffLogin(
  sdk: RecoverySDK,
  currentURL: string,
  replace: (_url: string) => void
) {
  if (sdk.isInClient()) {
    // login/logout are not supported here. Re-enter via the official LIFF URL.
    replace(await sdk.permanentLink.createUrlBy(currentURL))
  } else {
    // Reopening/reloading the same page otherwise retains the rejected token.
    if (sdk.isLoggedIn()) sdk.logout()
    sdk.login({ redirectUri: currentURL })
  }
}
