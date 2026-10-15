import type { NextConfig } from "next";
const config: NextConfig = {
  poweredByHeader: false,
  async rewrites() {
    const origin = process.env.NANOLM_BACKEND_ORIGIN;
    if (!origin) return [];
    const url = new URL(origin);
    if (url.protocol !== 'https:' || url.username || url.password || url.pathname !== '/' || url.search || url.hash) {
      throw new Error('NANOLM_BACKEND_ORIGIN must be an HTTPS origin without credentials or paths');
    }
    return [{ source: '/api/:path*', destination: `${url.origin}/:path*` }];
  },
};
export default config;
