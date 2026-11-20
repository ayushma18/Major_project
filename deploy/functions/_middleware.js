// Password gate for the whole site (Cloudflare Pages Functions middleware).
//
// One password, no user name. Anyone without a valid session cookie gets a login page
// (HTML requests) or a 401 (data and PDF requests). The password is the Pages secret
// SITE_PASSWORD; the session cookie is an HMAC of a fixed label keyed by that password, so
// changing the password logs everybody out. Only /web/fonts/ is public, so the login page
// can use the project typeface.

const COOKIE = "aa_session";
const MAX_AGE = 60 * 60 * 24 * 30; // 30 days
const PUBLIC_PREFIXES = ["/web/fonts/"];

const enc = new TextEncoder();

async function sha256(text) {
  return crypto.subtle.digest("SHA-256", enc.encode(text));
}

async function sameString(a, b) {
  const [x, y] = await Promise.all([sha256(a), sha256(b)]);
  return crypto.subtle.timingSafeEqual(x, y);
}

async function sessionToken(password) {
  const key = await crypto.subtle.importKey("raw", enc.encode(password), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode("affect-as-attack-session-v1"));
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function readCookie(header, name) {
  for (const part of (header || "").split(";")) {
    const [k, ...v] = part.trim().split("=");
    if (k === name) return v.join("=");
  }
  return "";
}

function safeNext(path, hash) {
  let p = typeof path === "string" && path.startsWith("/") && !path.startsWith("//") ? path : "/web/";
  if (p === "/" || p === "/index.html") p = "/web/";
  const h = typeof hash === "string" && /^#\/[\w\-\/.%:@]*$/.test(hash) ? hash : "";
  return p + h;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function loginPage(nextPath, failed) {
  const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>Affect as Attack</title>
<style>
  @font-face { font-family: "Schibsted Grotesk"; src: url("/web/fonts/SchibstedGrotesk-wght.woff") format("woff"); font-weight: 400 900; font-display: swap; }
  :root { --bg: #F2F2F0; --card: #FFFFFF; --ink: #0E0E0E; --muted: #5E5E5A; --line: #DCDCD7; --accent: #B3261E; --field: #F7F7F5; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #121212; --card: #1C1C1B; --ink: #F2F2F0; --muted: #A3A39E; --line: #33332F; --accent: #E5584F; --field: #232321; }
  }
  * { box-sizing: border-box; margin: 0; }
  html, body { height: 100%; }
  body { background: var(--bg); color: var(--ink); font-family: "Schibsted Grotesk", -apple-system, BlinkMacSystemFont, sans-serif;
         display: grid; place-items: center; padding: 24px; -webkit-font-smoothing: antialiased; }
  .card { width: 100%; max-width: 420px; background: var(--card); border: 1px solid var(--line); border-radius: 20px; padding: 40px 36px 36px; }
  .label { font-size: 12px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
  h1 { font-size: 34px; line-height: 1.1; font-weight: 800; letter-spacing: -0.01em; margin-top: 10px; }
  .lede { color: var(--muted); font-size: 15px; line-height: 1.55; margin-top: 12px; }
  form { margin-top: 28px; }
  .field { position: relative; }
  input[type=password] { width: 100%; font: inherit; font-size: 17px; color: var(--ink); background: var(--field);
         border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; outline: none; transition: border-color .15s, box-shadow .15s; }
  input[type=password]:focus { border-color: var(--ink); box-shadow: 0 0 0 3px color-mix(in srgb, var(--ink) 12%, transparent); }
  input[aria-invalid=true] { border-color: var(--accent); }
  .error { color: var(--accent); font-size: 14px; margin-top: 10px; }
  .error:empty { display: none; }
  button { margin-top: 18px; width: 100%; font: inherit; font-size: 16px; font-weight: 700; color: var(--card); background: var(--ink);
           border: 0; border-radius: 999px; padding: 14px 20px; cursor: pointer; transition: transform .1s, opacity .15s; }
  button:hover { opacity: 0.88; }
  button:active { transform: scale(0.99); }
  button:focus-visible { outline: 3px solid var(--accent); outline-offset: 3px; }
  .foot { margin-top: 26px; padding-top: 18px; border-top: 1px solid var(--line); color: var(--muted); font-size: 13px; line-height: 1.5; }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
</head>
<body>
  <main class="card">
    <p class="label">Project workspace</p>
    <h1>Affect as Attack</h1>
    <p class="lede">Emotion vectors versus the safety guardrail in Llama 3.1 8B. Papers, results and the proposal for the team.</p>
    <form method="post" action="/__login" autocomplete="on">
      <input type="hidden" name="next" value="${escapeHtml(nextPath)}">
      <input type="hidden" name="hash" id="hash" value="">
      <input type="text" name="username" value="team" autocomplete="username" hidden aria-hidden="true">
      <div class="field">
        <input type="password" name="password" id="password" placeholder="Password" aria-label="Password" autocomplete="current-password" required autofocus${failed ? ' aria-invalid="true"' : ""}>
      </div>
      <p class="error" role="alert">${failed ? "That password is not right. Try again." : ""}</p>
      <button type="submit">Open the workspace</button>
    </form>
    <p class="foot">BE major project, Pulchowk Campus. Anuj Paudel, Ashish Pandey, Ayushma Pudasaini.</p>
  </main>
  <script>document.getElementById("hash").value = location.hash || "";</script>
</body>
</html>`;
  return new Response(html, {
    status: 401,
    headers: { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store", "X-Robots-Tag": "noindex, nofollow" },
  });
}

export async function onRequest({ request, env, next }) {
  const password = env.SITE_PASSWORD;
  if (!password) {
    return new Response("Site password is not configured.", { status: 503, headers: { "Cache-Control": "no-store" } });
  }
  const url = new URL(request.url);
  if (PUBLIC_PREFIXES.some((p) => url.pathname.startsWith(p))) return next();

  const token = await sessionToken(password);

  if (url.pathname === "/__login" && request.method === "POST") {
    const form = await request.formData();
    const given = String(form.get("password") || "");
    const target = safeNext(String(form.get("next") || ""), String(form.get("hash") || ""));
    if (await sameString(given, password)) {
      return new Response(null, {
        status: 303,
        headers: { Location: target, "Cache-Control": "no-store", "Set-Cookie": `${COOKIE}=${token}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=${MAX_AGE}` },
      });
    }
    return loginPage(safeNext(String(form.get("next") || "")), true);
  }

  if (url.pathname === "/__logout") {
    return new Response(null, { status: 303, headers: { Location: "/web/", "Set-Cookie": `${COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0` } });
  }

  const cookie = readCookie(request.headers.get("Cookie"), COOKIE);
  if (cookie && (await sameString(cookie, token))) {
    const response = await next();
    const out = new Response(response.body, response);
    out.headers.set("X-Robots-Tag", "noindex, nofollow");
    out.headers.set("Cache-Control", "private, no-store");
    return out;
  }

  const wantsHtml = request.method === "GET" && (request.headers.get("Accept") || "").includes("text/html");
  if (wantsHtml) return loginPage(url.pathname + url.search, false);
  return new Response("Login required.", { status: 401, headers: { "Cache-Control": "no-store" } });
}
