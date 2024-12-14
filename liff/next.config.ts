import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  /* config options here */
  images: {
    remotePatterns: [{ hostname: 'profile.line-scdn.net' }] // Add the required hostname here
  }
}

export default nextConfig
