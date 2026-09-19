# LIFF review app

Learner-facing review interface opened from the LINE bot: registration and
profile, scores and daily-best trend, synchronized student/expert video
comparison, weekly review and notes, class averages, and the GPT coaching
history.

Routes: `/personal` (scores, 影片比較, 每週回顧), `/class`, `/gpt-chat` and
`/profile` (個人資料). The video tab offers both renders of the learner's own
stroke -- 教練建議版, with the coach's corrections written in and a freeze on
each, and 分析原片 without them -- and shows the attempt's thumbnail as the
player's poster while the video loads.

It is a Next.js static export (`output: 'export'`) served by Netlify. All data
comes from the Go backend in `../linebot`, authenticated with the learner's LINE
ID token (`Authorization: Bearer`) while it has more than a minute left, and with
the LIFF access token (`X-Line-Access-Token`) after that, so a page left open past
the ID token's hour keeps working. Learners register here, not in the chat: an
unregistered learner gets the registration form instead of the app, and the same
three fields -- experiment number, real name, handedness -- stay editable under
個人資料. Signed video URLs are re-fetched shortly before they expire.

## Development

```bash
npm ci
npm run dev      # http://localhost:3000
npm test         # node --test on src/**/*.test.ts
npm run build    # static export to out/, then writes out/_headers
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

`npm run build` runs `scripts/write-headers.mjs` as `postbuild`, which writes
`out/_headers` so Netlify serves a Content-Security-Policy (allowing the backend
origin from `NEXT_PUBLIC_BACKEND_BASE_URL`, the LIFF SDK's LINE hosts and signed
video URLs on `storage.googleapis.com`), HSTS, `nosniff`, a referrer policy and a
permissions policy on every path. The build fails if the backend URL is missing
or the file is not written. The policy lives in `src/lib/securityHeaders.ts`; add
any new external host there. No `frame-ancestors`/`X-Frame-Options` is sent,
because LINE clients may embed the LIFF view.
