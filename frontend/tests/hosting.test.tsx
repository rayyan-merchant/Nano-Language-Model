import { afterEach, expect, it, vi } from 'vitest';
import config from '../next.config';
const rewrites = config.rewrites as () => Promise<Array<{source: string; destination: string}>>;
afterEach(() => vi.unstubAllEnvs());
it('keeps local operation unchanged without a hosted upstream', async () => {
  vi.stubEnv('NANOLM_BACKEND_ORIGIN', '');
  expect(await rewrites()).toEqual([]);
});
it('proxies only the explicit API prefix to the configured HTTPS backend', async () => {
  vi.stubEnv('NANOLM_BACKEND_ORIGIN', 'https://example-api.vercel.app');
  expect(await rewrites()).toEqual([{source:'/api/:path*', destination:'https://example-api.vercel.app/:path*'}]);
});
it.each(['http://insecure.test', 'https://user:password@example.test', 'https://example.test/path'])('rejects invalid upstream %s', async (origin) => {
  vi.stubEnv('NANOLM_BACKEND_ORIGIN', origin);
  await expect(rewrites()).rejects.toThrow('must be an HTTPS origin');
});
