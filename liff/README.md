# LIFF review app

Learner-facing review interface opened from the LINE bot: scores and daily-best
trend, synchronized student/expert video comparison, weekly review and notes,
class averages, and the GPT coaching history.

It is a Next.js static export (`output: 'export'`) served by Netlify. All data
comes from the Go backend in `../linebot`, authenticated with the learner's LINE
ID token; learners register in the LINE chat before the app unlocks.

## Development

```bash
npm ci
npm run dev      # http://localhost:3000
npm test         # node --test on src/**/*.test.ts
npm run build    # static export to out/
```

Environment variables (build time, all public):

| Variable | Purpose |
|---|---|
| `NEXT_PUBLIC_LIFF_ID` | LIFF app ID; its prefix is the LINE Login channel the backend verifies tokens against |
| `NEXT_PUBLIC_BACKEND_BASE_URL` | Go backend base URL |
| `NEXT_PUBLIC_LIFF_REDIRECT_URI` | Optional LINE Login redirect override |
| `NEXT_PUBLIC_DEV_USER_ID`, `NEXT_PUBLIC_DEV_ID_TOKEN`, `NEXT_PUBLIC_DEV_DISPLAY_NAME` | Development only: bypass LIFF login locally. The backend still verifies the token, so a real LINE ID token is required |

## Deployment

`.github/workflows/cd-liff.yml` builds with the production backend URL on every
push to `main` that touches `liff/`, checks the URL was compiled in, and deploys
`out/` to Netlify.
