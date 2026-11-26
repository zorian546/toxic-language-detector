"use strict";

const element = (id) => document.getElementById(id);
const storageKey = "semantic-scholars-session-v1";
const emptySession = () => ({ checked: 0, flagged: 0, terms: {} });
let session = emptySession();
let busy = false;

try {
  const stored = JSON.parse(sessionStorage.getItem(storageKey));
  const isCount = (value) => Number.isSafeInteger(value) && value >= 0;
  if (stored && isCount(stored.checked) && isCount(stored.flagged) && stored.flagged <= stored.checked
      && stored.terms && typeof stored.terms === "object" && !Array.isArray(stored.terms)
      && Object.values(stored.terms).every(count => isCount(count) && count > 0)) {
    session = stored;
  }
} catch {
  // Analytics also work when browser storage is disabled.
}

function save() {
  try {
    sessionStorage.setItem(storageKey, JSON.stringify(session));
  } catch {
    // Keep the current session in memory.
  }
}

function renderSession() {
  element("checked").textContent = session.checked;
  element("flagged").textContent = session.flagged;
  element("rate").textContent = session.checked ? `${Math.round(100 * session.flagged / session.checked)}%` : "0%";
  const rows = Object.entries(session.terms).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 8);
  element("empty-chart").hidden = rows.length > 0;
  element("term-chart").replaceChildren(...rows.map(([term, count]) => {
    const row = document.createElement("li");
    const name = document.createElement("span");
    const track = document.createElement("span");
    const bar = document.createElement("span");
    const total = document.createElement("span");
    name.textContent = term;
    track.className = "track";
    track.setAttribute("aria-hidden", "true");
    bar.className = "bar";
    bar.style.width = `${100 * count / rows[0][1]}%`;
    total.textContent = count;
    track.append(bar);
    row.append(name, track, total);
    return row;
  }));
}

function renderPrediction(result) {
  element("verdict").textContent = result.flagged ? "Flagged for review" : "No flag";
  element("prediction").dataset.state = result.flagged ? "flagged" : "clear";
  const matches = Object.keys(result.matches);
  element("evidence").hidden = false;
  element("matches").textContent = matches.length ? `Reviewed terms · ${matches.join(", ")}` : "No reviewed terms matched.";
  const signals = element("signals");
  signals.hidden = result.model !== "naive_bayes" || result.signals.length === 0;
  signals.textContent = `Abuse-associated model features · ${result.signals.join(", ")}`;
  const label = (flagged) => flagged ? "flagged" : "no flag";
  element("comparison").hidden = false;
  element("model-results").textContent =
    `Rules · ${label(result.rules_flagged)}. Naive Bayes · ${label(result.nb_flagged)} (score ${result.score.toFixed(3)}, cutoff ${result.threshold.toFixed(2)}).`;
}

function clearPrediction() {
  delete element("prediction").dataset.state;
  const label = element("comment").value.trim() ? "Ready to analyze" : "Awaiting a comment";
  const verdict = element("verdict");
  if (verdict.textContent !== label) verdict.textContent = label;
  element("evidence").hidden = true;
  element("comparison").hidden = true;
  element("comparison").open = false;
}

async function analyze(comment) {
  const response = await fetch("/api/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: comment }),
  });
  if (!response.ok) {
    const issue = await response.json().catch(() => ({}));
    throw new Error(issue.error ?? "Unable to check the comment. Try again.");
  }
  const result = await response.json();
  session.checked += 1;
  session.flagged += Number(result.flagged);
  for (const [term, count] of Object.entries(result.matches)) {
    session.terms[term] = (Object.hasOwn(session.terms, term) ? session.terms[term] : 0) + count;
  }
  save();
  renderPrediction(result);
  renderSession();
}

async function run(comments) {
  if (busy) return;
  busy = true;
  const error = element("error");
  const field = element("comment");
  const analyzeButton = element("analyze");
  const activeButton = comments.length === 1 ? analyzeButton : element("samples");
  const controls = [analyzeButton, element("samples"), element("reset")];
  const label = activeButton.textContent;
  controls.forEach((button) => { button.setAttribute("aria-disabled", "true"); });
  activeButton.textContent = "Analyzing…";
  element("prediction").setAttribute("aria-busy", "true");
  field.readOnly = true;
  error.hidden = true;
  try {
    for (const comment of comments) {
      field.value = comment;
      clearPrediction();
      await analyze(comment);
    }
  } catch (failure) {
    error.textContent = failure.message;
    error.hidden = false;
  } finally {
    field.readOnly = false;
    activeButton.textContent = label;
    controls.forEach((button) => { button.removeAttribute("aria-disabled"); });
    element("prediction").removeAttribute("aria-busy");
    busy = false;
  }
}

element("comment-form").addEventListener("submit", (event) => {
  event.preventDefault();
  run([element("comment").value.trim()]);
});
element("samples").addEventListener("click", () => run([
  "Could you share the notes from today's meeting?",
  "You are a complete idiot. Shut up!",
  "Fuck off, you pathetic asshole!",
]));
element("reset").addEventListener("click", () => {
  if (busy) return;
  session = emptySession();
  save();
  renderSession();
});
element("comment").addEventListener("input", () => {
  element("error").hidden = true;
  clearPrediction();
});

const tabs = [element("show-jigsaw"), element("show-chat")];
function selectTab(selected) {
  for (const tab of tabs) {
    const active = tab === selected;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    element(tab.getAttribute("aria-controls")).hidden = !active;
  }
}
for (const [index, tab] of tabs.entries()) {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", (event) => {
    const keys = { ArrowLeft: 1 - index, ArrowRight: 1 - index, Home: 0, End: tabs.length - 1 };
    if (!Object.hasOwn(keys, event.key)) return;
    event.preventDefault();
    const next = tabs[keys[event.key]];
    selectTab(next);
    next.focus();
  });
}
renderSession();
