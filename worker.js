/* Clearcut worker: downloads + caches the model, runs it (WebGPU or WASM),
   then refines the mask at full resolution:
     1. removes stray specks
     2. guided filter: snaps the soft mask to real edges in the photo
     3. foreground colour estimation: removes background colour bleeding into hair/fur edges
*/
importScripts('ort/ort.webgpu.min.js');

const CACHE = 'clearcut-v3';
// Files are split into ~10 MB pieces so they fit GitHub's 25 MB web-upload limit.
const parts = (prefix, n) => Array.from({ length: n }, (_, i) => `${prefix}.part${i}.bin`);
const MODELS = {
  best: { parts: parts('models/isnet', 5), size: 1024,
          mean: [0.5, 0.5, 0.5], std: [1, 1, 1] },
  fast: { parts: ['models/u2netp.onnx'], size: 320,
          mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
};
const WASM_PARTS = parts('ort/engine', 3);

function join(chunks) {
  const len = chunks.reduce((a, c) => a + c.length, 0);
  const out = new Uint8Array(len); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}
async function fetchParts(urls, track) {
  const chunks = [];
  for (const u of urls) chunks.push(await cachedFetch(u, track));
  return join(chunks);
}
// drop files cached by older versions of the site
caches && caches.keys && caches.keys().then(ks => ks.forEach(k => { if (k.startsWith('clearcut-') && k !== CACHE) caches.delete(k); })).catch(() => {});

const sessions = {};      // name -> {session, backend}
let wasmReady = null;     // promise
let last = null;          // cached inputs for re-refining without re-running the model

const post = (m, t) => self.postMessage(m, t || []);

/* ---------- downloads with progress + Cache Storage ---------- */
async function cachedFetch(url, onBytes) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch (e) { cache = null; }
  if (cache) {
    try {
      const hit = await cache.match(url);
      if (hit) { const b = new Uint8Array(await hit.arrayBuffer()); onBytes(b.length, true); return b; }
    } catch (e) {}
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error('Could not download ' + url + ' (HTTP ' + r.status + ')');
  let buf;
  if (r.body && r.body.getReader) {
    const reader = r.body.getReader(); const parts = []; let got = 0;
    for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); got += value.length; onBytes(value.length, false); }
    buf = new Uint8Array(got); let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
  } else { buf = new Uint8Array(await r.arrayBuffer()); onBytes(buf.length, false); }
  if (cache) { try { await cache.put(url, new Response(buf.slice(0), { headers: { 'content-type': 'application/octet-stream' } })); } catch (e) {} }
  return buf;
}
async function headSize(url) {
  try { const c = await caches.open(CACHE); if (await c.match(url)) return 0; } catch (e) {}
  try { const r = await fetch(url, { method: 'HEAD' }); return +r.headers.get('content-length') || 0; } catch (e) { return 0; }
}

async function ensureWasm(track) {
  if (!wasmReady) wasmReady = (async () => {
    const bin = await fetchParts(WASM_PARTS, track);
    ort.env.wasm.wasmBinary = bin.buffer;
    ort.env.wasm.wasmPaths = new URL('ort/', self.location.href).href;
    const cores = (self.navigator && navigator.hardwareConcurrency) || 2;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, Math.floor(cores / 2))) : 1;
    ort.env.logLevel = 'error';
  })();
  return wasmReady;
}

async function hasWebGPU() {
  try {
    if (!self.navigator || !navigator.gpu) return false;
    const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    return !!a;
  } catch (e) { return false; }
}

async function createSession(bytes, backend) {
  return ort.InferenceSession.create(bytes, {
    executionProviders: [backend], graphOptimizationLevel: 'all',
  });
}

async function load(name, forceCpu) {
  if (sessions[name] && !(forceCpu && sessions[name].backend !== 'wasm')) return sessions[name];
  const spec = MODELS[name];
  const urls = wasmReady ? spec.parts.slice() : [...WASM_PARTS, ...spec.parts];
  const sizes = await Promise.all(urls.map(headSize));
  const total = sizes.reduce((a, b) => a + b, 0);
  let done = 0, fromCache = true;
  const track = (n, cached) => {
    if (!cached) { done += n; fromCache = false; }
    post({ type: 'progress', stage: 'download', done, total });
  };
  await ensureWasm(track);
  const bytes = await fetchParts(spec.parts, track);
  post({ type: 'progress', stage: 'start', fromCache });

  let backend = 'wasm', session = null;
  if (!forceCpu && await hasWebGPU()) {
    try { session = await createSession(bytes, 'webgpu'); backend = 'webgpu'; }
    catch (e) { session = null; }
  }
  if (!session) { session = await createSession(bytes, 'wasm'); backend = 'wasm'; }
  sessions[name] = { session, backend, bytes };
  return sessions[name];
}

/* ---------- inference ---------- */
function toTensor(rgba, S, mean, std) {
  const n = S * S, t = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    t[i] = (rgba[i * 4] / 255 - mean[0]) / std[0];
    t[n + i] = (rgba[i * 4 + 1] / 255 - mean[1]) / std[1];
    t[2 * n + i] = (rgba[i * 4 + 2] / 255 - mean[2]) / std[2];
  }
  return t;
}
async function infer(name, small) {
  const spec = MODELS[name];
  let s = await load(name);
  const S = spec.size;
  const feeds = {};
  const run = async () => {
    feeds[s.session.inputNames[0]] = new ort.Tensor('float32', toTensor(small, S, spec.mean, spec.std), [1, 3, S, S]);
    const out = await s.session.run(feeds);
    return out[s.session.outputNames[0]];
  };
  let t0 = performance.now(), res;
  try { res = await run(); }
  catch (e) {
    if (s.backend !== 'webgpu') throw e;
    // GPU ran out of memory or hit a driver limit: retry on the CPU.
    post({ type: 'progress', stage: 'fallback' });
    try { await s.session.release(); } catch (_) {}
    delete sessions[name];
    s = await load(name, true);
    t0 = performance.now(); res = await run();
  }
  const d = res.data instanceof Float32Array ? res.data : Float32Array.from(res.data);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < d.length; i++) { const v = d[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
  const span = hi - lo || 1, m = new Float32Array(S * S);
  for (let i = 0; i < m.length; i++) m[i] = (d[i] - lo) / span;
  if (res.dispose) try { res.dispose(); } catch (_) {}
  return { mask: m, S, ms: performance.now() - t0, backend: s.backend };
}

/* ---------- 1. stray specks ---------- */
function removeSpecks(m, S) {
  const n = S * S, lab = new Int32Array(n), sizes = [0], stack = new Int32Array(n);
  let next = 1;
  for (let i = 0; i < n; i++) {
    if (lab[i] || m[i] <= 0.1) continue;
    let sp = 0, count = 0; stack[sp++] = i; lab[i] = next;
    while (sp) {
      const p = stack[--sp]; count++;
      const x = p % S, y = (p - x) / S;
      if (x > 0 && !lab[p - 1] && m[p - 1] > 0.1) { lab[p - 1] = next; stack[sp++] = p - 1; }
      if (x < S - 1 && !lab[p + 1] && m[p + 1] > 0.1) { lab[p + 1] = next; stack[sp++] = p + 1; }
      if (y > 0 && !lab[p - S] && m[p - S] > 0.1) { lab[p - S] = next; stack[sp++] = p - S; }
      if (y < S - 1 && !lab[p + S] && m[p + S] > 0.1) { lab[p + S] = next; stack[sp++] = p + S; }
    }
    sizes.push(count); next++;
  }
  if (next <= 2) return m;
  const largest = Math.max(...sizes);
  const minKeep = Math.max(n * 0.0015, largest * 0.04);
  const out = new Float32Array(m);
  for (let i = 0; i < n; i++) if (lab[i] && sizes[lab[i]] < minKeep) out[i] = 0;
  // also clear faint haze (<0.1) that isn't next to a kept region
  return out;
}

/* ---------- helpers ---------- */
function upsample(m, S, W, H) {
  const out = new Float32Array(W * H), sx = S / W, sy = S / H;
  for (let y = 0; y < H; y++) {
    let fy = (y + 0.5) * sy - 0.5; if (fy < 0) fy = 0;
    const y0 = Math.min(S - 1, fy | 0), y1 = Math.min(S - 1, y0 + 1), wy = fy - y0;
    for (let x = 0; x < W; x++) {
      let fx = (x + 0.5) * sx - 0.5; if (fx < 0) fx = 0;
      const x0 = Math.min(S - 1, fx | 0), x1 = Math.min(S - 1, x0 + 1), wx = fx - x0;
      const a = m[y0 * S + x0], b = m[y0 * S + x1], c = m[y1 * S + x0], d = m[y1 * S + x1];
      out[y * W + x] = (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy;
    }
  }
  return out;
}
// separable box mean with edge-correct normalisation; src may alias dst
function boxBlur(src, dst, W, H, r, tmp) {
  for (let y = 0; y < H; y++) {
    const o = y * W; let s = 0;
    const lim = Math.min(r, W - 1);
    for (let x = 0; x <= lim; x++) s += src[o + x];
    for (let x = 0; x < W; x++) {
      const l = x - r - 1, rr = x + r;
      if (x > 0) { if (rr < W) s += src[o + rr]; if (l >= 0) s -= src[o + l]; }
      const cnt = Math.min(W - 1, rr) - Math.max(0, x - r) + 1;
      tmp[o + x] = s / cnt;
    }
  }
  for (let x = 0; x < W; x++) {
    let s = 0; const lim = Math.min(r, H - 1);
    for (let y = 0; y <= lim; y++) s += tmp[y * W + x];
    for (let y = 0; y < H; y++) {
      const t = y - r - 1, b = y + r;
      if (y > 0) { if (b < H) s += tmp[b * W + x]; if (t >= 0) s -= tmp[t * W + x]; }
      const cnt = Math.min(H - 1, b) - Math.max(0, y - r) + 1;
      dst[y * W + x] = s / cnt;
    }
  }
  return dst;
}

/* ---------- 2. guided filter (grey guide) ---------- */
function guidedFilter(rgba, p, W, H, r, eps) {
  const n = W * H, I = new Float32Array(n);
  for (let i = 0; i < n; i++) I[i] = (0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2]) / 255;
  const tmp = new Float32Array(n), mI = new Float32Array(n), mP = new Float32Array(n), A = new Float32Array(n), B = new Float32Array(n);
  boxBlur(I, mI, W, H, r, tmp);
  boxBlur(p, mP, W, H, r, tmp);
  for (let i = 0; i < n; i++) { A[i] = I[i] * p[i]; B[i] = I[i] * I[i]; }
  boxBlur(A, A, W, H, r, tmp);   // mean(I*p)
  boxBlur(B, B, W, H, r, tmp);   // mean(I*I)
  for (let i = 0; i < n; i++) {
    const cov = A[i] - mI[i] * mP[i], v = B[i] - mI[i] * mI[i];
    const a = cov / (v + eps);
    A[i] = a; B[i] = mP[i] - a * mI[i];
  }
  boxBlur(A, A, W, H, r, tmp);
  boxBlur(B, B, W, H, r, tmp);
  const q = mP; // reuse
  for (let i = 0; i < n; i++) {
    let v = A[i] * I[i] + B[i];
    // only let the filter move the mask near edges; keep confident areas as the model said
    const o = p[i];
    if (o > 0.97 && v < o) v = o - (o - v) * 0.25;
    if (o < 0.03 && v > o) v = o + (v - o) * 0.25;
    q[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return q;
}

/* ---------- 3. foreground colour estimation (blur fusion, Forte & Pitié 2021) ---------- */
function estimateForeground(rgba, alpha, W, H) {
  const n = W * H, maxSide = Math.max(W, H);
  const r1 = Math.max(10, Math.round(90 * maxSide / 2000)), r2 = Math.max(2, Math.round(6 * maxSide / 2000));
  const tmp = new Float32Array(n), bA1 = new Float32Array(n), bA2 = new Float32Array(n);
  boxBlur(alpha, bA1, W, H, r1, tmp);
  boxBlur(alpha, bA2, W, H, r2, tmp);
  const I = new Float32Array(n), F = new Float32Array(n), Bk = new Float32Array(n), t1 = new Float32Array(n), t2 = new Float32Array(n);
  const out = new Uint8ClampedArray(rgba); // starts as the original
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < n; i++) I[i] = rgba[i * 4 + c] / 255;
    // stage 1: F = B = I, wide radius
    for (let i = 0; i < n; i++) { t1[i] = I[i] * alpha[i]; t2[i] = I[i] * (1 - alpha[i]); }
    boxBlur(t1, t1, W, H, r1, tmp); boxBlur(t2, t2, W, H, r1, tmp);
    for (let i = 0; i < n; i++) {
      const a = alpha[i], bf = t1[i] / (bA1[i] + 1e-5), bb = t2[i] / (1 - bA1[i] + 1e-5);
      let f = bf + a * (I[i] - a * bf - (1 - a) * bb);
      F[i] = f < 0 ? 0 : f > 1 ? 1 : f; Bk[i] = bb;
    }
    // stage 2: refine with a small radius
    for (let i = 0; i < n; i++) { t1[i] = F[i] * alpha[i]; t2[i] = Bk[i] * (1 - alpha[i]); }
    boxBlur(t1, t1, W, H, r2, tmp); boxBlur(t2, t2, W, H, r2, tmp);
    for (let i = 0; i < n; i++) {
      const a = alpha[i];
      if (a <= 0.01 || a >= 0.99) continue; // keep original pixels where the mask is certain
      const bf = t1[i] / (bA2[i] + 1e-5), bb = t2[i] / (1 - bA2[i] + 1e-5);
      const f = bf + a * (I[i] - a * bf - (1 - a) * bb);
      out[i * 4 + c] = f * 255;
    }
  }
  for (let i = 0; i < n; i++) out[i * 4 + 3] = 255;
  return out;
}

function refine(opts) {
  const { coarse, S, rgba, W, H, cleanup, edges } = opts;
  const t0 = performance.now();
  const m = cleanup ? removeSpecks(coarse, S) : coarse;
  let alpha = upsample(m, S, W, H);
  if (edges) {
    const scale = Math.max(W, H) / S;
    const r = Math.max(2, Math.round(2 * Math.max(1, scale)));
    alpha = guidedFilter(rgba, alpha, W, H, r, 1e-3);
  }
  const fg = edges ? estimateForeground(rgba, alpha, W, H) : new Uint8ClampedArray(rgba);
  return { alpha, fg, ms: performance.now() - t0 };
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'load') {
      const s = await load(msg.model);
      post({ type: 'loaded', model: msg.model, backend: s.backend });
    } else if (msg.type === 'run') {
      post({ type: 'progress', stage: 'infer' });
      const r = await infer(msg.model, msg.small);
      last = { coarse: r.mask, S: r.S, rgba: msg.rgba, W: msg.W, H: msg.H };
      post({ type: 'progress', stage: 'refine' });
      const f = refine({ ...last, cleanup: msg.cleanup, edges: msg.edges });
      post({ type: 'result', id: msg.id, alpha: f.alpha, fg: f.fg, backend: r.backend, inferMs: r.ms, refineMs: f.ms },
           [f.alpha.buffer, f.fg.buffer]);
    } else if (msg.type === 'refine') {
      if (!last) return;
      const f = refine({ ...last, cleanup: msg.cleanup, edges: msg.edges });
      post({ type: 'result', id: msg.id, alpha: f.alpha, fg: f.fg, refineMs: f.ms, refineOnly: true },
           [f.alpha.buffer, f.fg.buffer]);
    }
  } catch (err) {
    post({ type: 'error', id: msg.id, message: String(err && err.message || err) });
  }
};
