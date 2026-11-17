/**
 * RoadVision AI — centralised API base URL helper.
 *
 * In development (Vite dev server) the proxy forwards bare paths like
 * /api/... and /predict to the local backend, so an empty base works fine.
 *
 * In production (Vercel static build) VITE_API_BASE_URL must be set to the
 * Render backend URL, e.g. https://roadvision-ai.onrender.com
 * All fetch calls go to that origin directly.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL
  ? import.meta.env.VITE_API_BASE_URL.replace(/\/$/, '') // strip trailing slash
  : '';

/**
 * Build an absolute URL for a given API path.
 * @param {string} path - must start with /
 */
export function apiUrl(path) {
  return `${API_BASE}${path}`;
}

export default API_BASE;
