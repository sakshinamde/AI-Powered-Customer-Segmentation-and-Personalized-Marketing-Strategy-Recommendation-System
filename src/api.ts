const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
const AUTH_TOKEN_KEY = 'smartseg-access-token';

export function setAccessToken(token: string | null): void {
  if (token) {
    localStorage.setItem(AUTH_TOKEN_KEY, token);
  } else {
    localStorage.removeItem(AUTH_TOKEN_KEY);
  }
}

export async function apiFetch(path: string, options: RequestInit = {}): Promise<Response> {
  const headers = new Headers(options.headers);
  const token = localStorage.getItem(AUTH_TOKEN_KEY);
  if (token) headers.set('Authorization', `Bearer ${token}`);
  return fetch(`${API_BASE_URL}${path}`, { ...options, headers });
}
