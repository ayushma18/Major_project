// Password gate for the whole site (Cloudflare Pages Functions middleware).
// Every request (pages, JSON data, PDFs) needs HTTP Basic Auth. Any user name works;
// the password is the Pages secret SITE_PASSWORD (set with `wrangler pages secret put`).
// The comparison hashes both values and uses a constant-time compare.

const REALM = 'Basic realm="Affect as Attack project", charset="UTF-8"';

async function sha256(text) {
  return crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
}

async function passwordMatches(given, expected) {
  const [a, b] = await Promise.all([sha256(given), sha256(expected)]);
  return crypto.subtle.timingSafeEqual(a, b);
}

function denied() {
  return new Response("Password required.", {
    status: 401,
    headers: { "WWW-Authenticate": REALM, "Cache-Control": "no-store", "X-Robots-Tag": "noindex" },
  });
}

export async function onRequest({ request, env, next }) {
  const expected = env.SITE_PASSWORD;
  if (!expected) {
    return new Response("Site password is not configured.", { status: 503, headers: { "Cache-Control": "no-store" } });
  }
  const header = request.headers.get("Authorization") || "";
  if (!header.startsWith("Basic ")) return denied();
  let decoded = "";
  try {
    decoded = atob(header.slice(6));
  } catch {
    return denied();
  }
  const password = decoded.slice(decoded.indexOf(":") + 1);
  if (!(await passwordMatches(password, expected))) return denied();
  const response = await next();
  const out = new Response(response.body, response);
  out.headers.set("X-Robots-Tag", "noindex, nofollow");
  out.headers.set("Cache-Control", "private, no-store");
  return out;
}
