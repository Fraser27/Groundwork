/* ─────────────────────────────────────────────────────────────────────────────
   Shell: sidebar, theme toggle, on-page table of contents, prev/next.

   Every page is a complete HTML file so it can be opened directly from disk.
   The sidebar is rendered from the PAGES array below rather than copied into
   twelve files, so adding a page means editing one list.

   No fetch(), no modules, no build step — a browser reading file:// will not
   fetch a sibling JSON file, and it will not load an ES module from disk
   either. Everything here is a plain script.
   ──────────────────────────────────────────────────────────────────────────── */

var PAGES = [
  { group: 'Start here', items: [
    { file: 'index.html', title: 'Overview' },
    { file: 'assertion-contract.html', title: 'The assertion contract' },
  ]},
  { group: 'How it works', items: [
    { file: 'extraction.html', title: 'How extraction works' },
    { file: 'provenance.html', title: 'Provenance' },
    { file: 'tenancy.html', title: 'Tenancy and ethical walls' },
    { file: 'predicates.html', title: 'The predicate vocabulary' },
  ]},
  { group: 'Using it', items: [
    { file: 'asking-questions.html', title: 'Asking questions' },
    { file: 'review.html', title: 'Reviewing claims' },
    { file: 'governance.html', title: 'Governance settings' },
    { file: 'demo-data.html', title: 'Loading the demo data' },
  ]},
  { group: 'Reference', items: [
    { file: 'glossary.html', title: 'Glossary' },
    { file: 'architecture.html', title: 'Architecture' },
  ]},
];

/** Flat page order, for prev/next. */
var FLAT = PAGES.reduce(function (acc, g) { return acc.concat(g.items); }, []);

function currentFile() {
  var path = window.location.pathname;
  var name = path.substring(path.lastIndexOf('/') + 1);
  return name === '' ? 'index.html' : name;
}

// ── Theme ──────────────────────────────────────────────────────────────────
//
// localStorage is unavailable or throws in some file:// contexts, so every
// access is guarded. A page that cannot remember the theme still toggles.

function readStoredTheme() {
  try { return window.localStorage.getItem('groundwork-docs-theme'); } catch (e) { return null; }
}

function storeTheme(theme) {
  try { window.localStorage.setItem('groundwork-docs-theme', theme); } catch (e) { /* ignore */ }
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  var btn = document.getElementById('theme-toggle');
  if (btn) {
    btn.textContent = theme === 'dark' ? '☀' : '☽';
    btn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
    btn.setAttribute('title', theme === 'dark' ? 'Light theme' : 'Dark theme');
  }
}

function initialTheme() {
  var stored = readStoredTheme();
  if (stored === 'dark' || stored === 'light') return stored;
  var prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  return prefersDark ? 'dark' : 'light';
}

// Applied before first paint by an inline call in each page's <head>, so the
// page does not flash the wrong theme.
window.__groundworkApplyTheme = function () { applyTheme(initialTheme()); };

function toggleTheme() {
  var next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  storeTheme(next);
}

// ── Sidebar ────────────────────────────────────────────────────────────────

function headingsOfPage() {
  var out = [];
  var nodes = document.querySelectorAll('main h2[id]');
  for (var i = 0; i < nodes.length; i++) {
    out.push({ id: nodes[i].id, text: nodes[i].textContent.trim() });
  }
  return out;
}

function renderSidebar() {
  var host = document.getElementById('sidebar');
  if (!host) return;
  var here = currentFile();

  var html = '';
  html += '<div class="sidebar-head">';
  html += '<div><h1><a href="index.html"><span class="mark">Lex</span>Graph</a></h1>';
  html += '<span class="sub">Documentation</span></div>';
  html += '<button id="theme-toggle" class="theme-toggle" type="button">☽</button>';
  html += '</div>';

  html += '<div class="search-wrap">';
  html += '<input id="search-box" class="search-box" type="search" autocomplete="off" ';
  html += 'spellcheck="false" placeholder="Search the docs" aria-label="Search the documentation">';
  html += '<span class="search-hint">/</span>';
  html += '</div>';

  html += '<nav aria-label="Documentation">';
  for (var g = 0; g < PAGES.length; g++) {
    html += '<div class="nav-group">' + PAGES[g].group + '</div>';
    for (var i = 0; i < PAGES[g].items.length; i++) {
      var item = PAGES[g].items[i];
      var isHere = item.file === here;
      html += '<a href="' + item.file + '"' + (isHere ? ' class="current" aria-current="page"' : '') + '>';
      html += item.title + '</a>';

      // The current page's own h2s are listed inline, so the sidebar doubles as
      // an on-page table of contents without a second column.
      if (isHere) {
        var hs = headingsOfPage();
        if (hs.length) {
          html += '<div class="toc">';
          for (var h = 0; h < hs.length; h++) {
            html += '<a href="#' + hs[h].id + '">' + hs[h].text + '</a>';
          }
          html += '</div>';
        }
      }
    }
  }
  html += '</nav>';

  host.innerHTML = html;
  document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
  applyTheme(document.documentElement.getAttribute('data-theme') || initialTheme());
}

// ── Prev / next ────────────────────────────────────────────────────────────

function renderPageNav() {
  var main = document.querySelector('main');
  if (!main) return;
  var here = currentFile();
  var idx = -1;
  for (var i = 0; i < FLAT.length; i++) { if (FLAT[i].file === here) idx = i; }
  if (idx < 0) return;

  var prev = idx > 0 ? FLAT[idx - 1] : null;
  var next = idx < FLAT.length - 1 ? FLAT[idx + 1] : null;
  if (!prev && !next) return;

  var nav = document.createElement('div');
  nav.className = 'page-nav';
  var html = '';
  if (prev) html += '<a href="' + prev.file + '"><span class="dir">Previous</span>' + prev.title + '</a>';
  if (next) html += '<a class="next" href="' + next.file + '"><span class="dir">Next</span>' + next.title + '</a>';
  nav.innerHTML = html;
  main.appendChild(nav);
}

// ── Glossary filter ────────────────────────────────────────────────────────
//
// Only the glossary page has one. Lives here rather than in a twelfth file.

function initGlossaryFilter() {
  var input = document.getElementById('glossary-filter');
  if (!input) return;

  // Every definition list on the page, not just the first — the glossary has a second
  // list for the ingest stages, and filtering only the first left those visible under
  // a query that matched nothing, which reads as though they were matches.
  var dts = document.querySelectorAll('dl.glossary dt');
  if (!dts.length) return;

  var terms = [];
  for (var i = 0; i < dts.length; i++) {
    var dds = [];
    var node = dts[i].nextElementSibling;
    while (node && node.tagName === 'DD') { dds.push(node); node = node.nextElementSibling; }
    terms.push({
      dt: dts[i],
      dds: dds,
      // Term and its definitions, matched together: someone searching "wall" should
      // find "Tenant" if its text explains walls.
      haystack: (dts[i].textContent + ' ' + dds.map(function (d) {
        return d.textContent;
      }).join(' ')).toLowerCase()
    });
  }

  input.addEventListener('input', function () {
    var q = input.value.trim().toLowerCase();
    for (var i = 0; i < terms.length; i++) {
      var t = terms[i];
      var show = !q || t.haystack.indexOf(q) >= 0;
      t.dt.style.display = show ? '' : 'none';
      for (var j = 0; j < t.dds.length; j++) t.dds[j].style.display = show ? '' : 'none';
    }
  });
}

document.addEventListener('DOMContentLoaded', function () {
  renderSidebar();
  renderPageNav();
  initGlossaryFilter();
});
