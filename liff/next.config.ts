import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  /* config options here */
  eslint: {
    // Allow production builds to succeed even if there are ESLint errors
    ignoreDuringBuilds: true
  },
  images: {
    remotePatterns: [{ hostname: 'profile.line-scdn.net' }] // Add the required hostname here
  }
}

export default nextConfig
