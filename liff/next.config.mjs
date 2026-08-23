/**
 * Plain ESM rather than TypeScript on purpose: a .ts config makes Next
 * transpile it through swc before the server starts, and that path is fragile
 * (it fails with "transform is not a function" when the swc binding resolves
 * oddly, even though the binary is fine). The config has no types worth the
 * risk of not being able to start the dev server.
 *
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  output: 'export',
  eslint: {
    // Allow production builds to succeed even if there are ESLint errors
    ignoreDuringBuilds: true
  },
  images: {
    remotePatterns: [{ hostname: 'profile.line-scdn.net' }], // Add the required hostname here
    unoptimized: true
  }
}

export default nextConfig
