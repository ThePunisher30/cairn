"use strict";

// The page: send a question, read the server-sent events, show the answer with citation chips
// that open the cited lines. Everything from the server (model text, file text) is untrusted and
// is put on the page with textContent, never as HTML.

const $ = (id) => document.getElementById(id);
const form = $("ask-form");
const input = $("question");
const button = $("ask-button");
const resultBox = $("result");
const answerBox = $("answer");
const refsBox = $("refs");
const passagesBox = $("passages");
const noticeBox = $("notice");
const footer = $("footer");
const statusLine = $("status");
const viewer = $("viewer");
const keyRow = $("key-row");
const keyInput = $("api-key");

let controller = null;
let authRequired = false;

function readKey() {
  try { return sessionStorage.getItem("cairn-key") || ""; } catch (e) { return ""; }
}
function saveKey(value) {
  try { sessionStorage.setItem("cairn-key", value); } catch (e) { /* private window: keep it in memory */ }
}
function headers(extra) {
  const h = Object.assign({}, extra || {});
  const key = keyInput.value.trim();
  if (key) h["X-API-Key"] = key;
  return h;
}

// --- server-sent events -------------------------------------------------------------------

// One block of text such as "event: token\ndata: {...}" becomes {event, data}.
function parseEvent(block) {
  let event = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return null;
  try { return { event, data: JSON.parse(data.join("\n")) }; } catch (e) { return null; }
}

async function* events(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const parsed = parseEvent(buffer.slice(0, cut));
      buffer = buffer.slice(cut + 2);
      if (parsed) yield parsed;
    }
  }
}

// --- showing things -----------------------------------------------------------------------

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

function notice(text) {
  noticeBox.hidden = !text;
  noticeBox.textContent = text || "";
}

// The answer text with each [n] or [1, 2] marker turned into a button that opens the source.
function renderAnswer(text, references) {
  clear(answerBox);
  const byNumber = new Map(references.map((r) => [r.number, r]));
  const marker = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  let last = 0;
  let match;
  while ((match = marker.exec(text)) !== null) {
    const numbers = match[1].split(",").map((n) => parseInt(n, 10));
    if (!numbers.every((n) => byNumber.has(n))) continue;
    answerBox.append(document.createTextNode(text.slice(last, match.index)));
    numbers.forEach((n, i) => {
      if (i > 0) answerBox.append(document.createTextNode(" "));
      answerBox.append(citeButton(byNumber.get(n), "[" + n + "]", "cite"));
    });
    last = match.index + match[0].length;
  }
  answerBox.append(document.createTextNode(text.slice(last)));
}

function lines(ref) {
  return ref.start_line === ref.end_line ? String(ref.start_line) : ref.start_line + "-" + ref.end_line;
}

function citeButton(ref, label, cls) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = cls;
  b.textContent = label;
  b.title = ref.path + ":" + lines(ref);
  b.addEventListener("click", () => openSource(ref));
  return b;
}

function renderRefs(references) {
  clear(refsBox);
  for (const ref of references) {
    refsBox.append(citeButton(ref, "[" + ref.number + "] " + ref.path + ":" + lines(ref), "chip"));
  }
}

function renderPassages(chunks, references) {
  clear(passagesBox);
  passagesBox.hidden = chunks.length === 0;
  chunks.forEach((chunk, i) => {
    const li = document.createElement("li");
    const where = document.createElement("div");
    where.className = "where";
    const ref = references[i];
    if (ref) where.append(citeButton(ref, ref.path + ":" + lines(ref), "chip"));
    const pre = document.createElement("pre");
    pre.textContent = chunk.text;
    li.append(where, pre);
    passagesBox.append(li);
  });
}

function renderFooter(done, clientMs, firstMs) {
  const parts = [];
  if (done.cached) parts.push("from the " + done.cached + " cache");
  if (firstMs != null) parts.push("first text after " + Math.round(firstMs) + " ms");
  parts.push("total " + Math.round(clientMs) + " ms in the browser");
  if (done.input_tokens || done.output_tokens) {
    parts.push(done.input_tokens + " tokens in, " + done.output_tokens + " out");
    parts.push("$" + done.cost_usd.toFixed(4) + " (estimated)");
  }
  if (done.model && done.model !== "none") parts.push(done.model);
  footer.textContent = parts.join(" · ");
}

// --- the source viewer --------------------------------------------------------------------

async function openSource(ref) {
  $("viewer-title").textContent = ref.path + ":" + lines(ref);
  const link = $("viewer-link");
  link.href = ref.url;
  const errorBox = $("viewer-error");
  const code = $("viewer-code");
  errorBox.hidden = true;
  clear(code);
  if (!viewer.open) viewer.showModal();
  try {
    const params = new URLSearchParams({ path: ref.path, start: ref.start_line, end: ref.end_line });
    const response = await fetch("/source?" + params, { headers: headers() });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    const source = await response.json();
    let firstCited = null;
    for (const line of source.lines) {
      const row = document.createElement("span");
      row.className = "ln" + (line.cited ? " cited" : "");
      const num = document.createElement("span");
      num.className = "num";
      num.textContent = line.n;
      row.append(num, document.createTextNode(line.text));
      code.append(row);
      if (line.cited && !firstCited) firstCited = row;
    }
    if (firstCited) code.scrollTop = Math.max(0, firstCited.offsetTop - 60);
  } catch (e) {
    errorBox.textContent = "Could not load the source here (" + e.message + "). The GitHub link still works.";
    errorBox.hidden = false;
  }
}

$("viewer-close").addEventListener("click", () => viewer.close());
viewer.addEventListener("click", (event) => { if (event.target === viewer) viewer.close(); });

// --- asking -------------------------------------------------------------------------------

async function ask(question) {
  if (controller) controller.abort();
  controller = new AbortController();
  const started = performance.now();
  let firstMs = null;
  let text = "";
  let references = [];

  button.disabled = true;
  resultBox.hidden = false;
  notice("");
  clear(answerBox);
  answerBox.classList.remove("idk");
  clear(refsBox);
  clear(passagesBox);
  passagesBox.hidden = true;
  footer.textContent = "";
  answerBox.textContent = "Searching...";

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ question, stream: true }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      if (response.status === 401) { keyRow.hidden = false; keyInput.focus(); }
      const wait = response.headers.get("Retry-After");
      throw new Error((detail.detail || response.statusText) + (wait ? " (try again in " + wait + " s)" : ""));
    }
    for await (const { event, data } of events(response)) {
      if (event === "token") {
        if (firstMs === null) { firstMs = performance.now() - started; clear(answerBox); }
        text += data.text;
        answerBox.textContent = text;
      } else if (event === "fallback") {
        clear(answerBox);
        notice(data.message);
      } else if (event === "error") {
        notice(data.message);
      } else if (event === "done") {
        references = data.references || [];
        if (data.degraded) {
          // no answer text: show what retrieval found
          answerBox.textContent = data.chunks.length ? "" : "Nothing was retrieved for that question.";
          notice(data.error);
          renderPassages(data.chunks, references);
        } else if (data.abstained) {
          answerBox.textContent = "I don't know: the retrieved passages do not answer this.";
          answerBox.classList.add("idk");
        } else {
          renderAnswer(data.answer, references);
          if (data.truncated) notice("The answer was cut off at the length limit.");
        }
        if (!data.degraded) renderRefs(references);
        renderFooter(data, performance.now() - started, firstMs);
      }
    }
  } catch (e) {
    if (e.name === "AbortError") return;
    clear(answerBox);
    notice("Could not get an answer: " + e.message);
  } finally {
    button.disabled = false;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (question) ask(question);
});

for (const example of document.querySelectorAll(".example")) {
  example.addEventListener("click", () => { input.value = example.textContent; input.focus(); ask(example.textContent); });
}

keyInput.addEventListener("change", () => saveKey(keyInput.value.trim()));

// --- the status line ----------------------------------------------------------------------

async function showStatus() {
  try {
    const response = await fetch("/health");
    const health = await response.json();
    authRequired = Boolean(health.auth_required);
    if (authRequired) { keyRow.hidden = false; keyInput.value = readKey(); }
    if (!health.ready) {
      statusLine.textContent = health.status === "failed" ? "The service failed to start." : "The service is starting (" + health.status + ")...";
      setTimeout(showStatus, 3000);
      return;
    }
    const mode = health.mode || {};
    const parts = [];
    parts.push(mode.llm === "none" ? "Retrieval only: no language model is attached, so you get the passages" : "Answers by " + mode.model);
    parts.push(mode.retriever === "bm25" ? "keyword search (BM25)" : "hybrid search (BM25 + embeddings)");
    if (health.index) parts.push(health.index.chunks + " passages");
    statusLine.textContent = parts.join(" · ");
  } catch (e) {
    statusLine.textContent = "Cannot reach the server.";
  }
}

showStatus();
