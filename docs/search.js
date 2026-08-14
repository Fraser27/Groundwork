/* ─────────────────────────────────────────────────────────────────────────────
   Client-side search.

   The index is a plain script (`search-index.js`) that assigns a global, not a
   `fetch()` of `search-index.json`. That is deliberate: a page opened from
   file:// is treated as an opaque origin, so fetching a sibling JSON file fails
   the CORS check in Chrome and Safari with no way to allow it. A <script> tag
   has no such restriction. `search-index.json` is still generated and shipped —
   it is the readable, diffable copy, and any tooling that wants the data should
   read that — but the browser gets the .js.

   Matching is substring plus token, in that order of confidence:

     1. the whole query appears in the title           strongest
     2. the whole query appears in a heading
     3. the whole query appears in the body
     4. every token appears somewhere on the page      weakest, but catches
                                                       word-order differences

   No stemming and no fuzzy matching. A lawyer searching "ethical wall" wants
   the ethical wall page, and a search that also returns four near-misses is
   worse than one that returns nothing.
   ──────────────────────────────────────────────────────────────────────────── */

(function () {
  'use strict';

  var MAX_RESULTS = 24;
  var SNIPPET_RADIUS = 62;

  var docs = window.LEXGRAPH_SEARCH_INDEX || [];
  var box = null;
  var overlay = null;
  var panel = null;
  var hits = [];
  var activeIndex = -1;

  // ── Index preparation ────────────────────────────────────────────────────
  //
  // Lowercased copies are cached once. The index is a few tens of kilobytes, so
  // this is cheaper than lowercasing on every keystroke.

  var entries = docs.map(function (d) {
    var headings = d.headings || [];
    return {
      path: d.path,
      title: d.title,
      summary: d.summary || '',
      body: d.body || '',
      headings: headings,
      lcTitle: (d.title || '').toLowerCase(),
      lcBody: (d.body || '').toLowerCase(),
      // `at` is the heading's offset into the body, recorded at build time. It is not
      // recomputed here by searching for the heading text: a heading's own words
      // usually appear earlier on the page as a cross-reference link, so a search
      // would find the link and put the section boundaries in the wrong order.
      lcHeadings: headings.map(function (h) {
        return { id: h.id, text: h.text, at: h.at || 0, lc: (h.text || '').toLowerCase() };
      })
    };
  });

  function tokenise(q) {
    return q.toLowerCase().split(/[^a-z0-9_]+/).filter(function (t) { return t.length > 1; });
  }

  /**
   * Score one page against a query. Returns null for a miss.
   *
   * `where` records what matched so the result can be pointed at a heading
   * anchor rather than the top of the page — the difference between "this page
   * mentions it" and "here it is".
   */
  function score(entry, query, tokens) {
    var lcq = query.toLowerCase();
    var best = null;

    if (entry.lcTitle.indexOf(lcq) >= 0) {
      best = { points: 100, where: null, at: -1 };
    }

    for (var i = 0; i < entry.lcHeadings.length; i++) {
      if (entry.lcHeadings[i].lc.indexOf(lcq) >= 0) {
        var headingScore = { points: 70 - i * 0.1, where: entry.lcHeadings[i], at: -1 };
        if (!best || headingScore.points > best.points) best = headingScore;
        break;
      }
    }

    var bodyAt = entry.lcBody.indexOf(lcq);
    if (bodyAt >= 0) {
      var bodyScore = { points: 40, where: nearestHeading(entry, bodyAt), at: bodyAt };
      if (!best || bodyScore.points > best.points) best = bodyScore;
    }

    if (!best && tokens.length > 1) {
      // Every token must be present somewhere. Weakest tier, so it only runs when the
      // phrase itself was not found anywhere — "cap floor" finds the cap/floor section
      // even though nobody writes those two words adjacent.
      var firstAt = -1;
      for (var t = 0; t < tokens.length; t++) {
        var at = entry.lcBody.indexOf(tokens[t]);
        var inTitle = entry.lcTitle.indexOf(tokens[t]) >= 0;
        if (at < 0 && !inTitle) return null;
        if (at >= 0 && (firstAt < 0 || at < firstAt)) firstAt = at;
      }

      // A heading containing every token is a much better answer than the first
      // scattered mention, so it both wins the ranking and supplies the anchor.
      var headingHit = null;
      for (var hh = 0; hh < entry.lcHeadings.length && !headingHit; hh++) {
        var all = true;
        for (var tt = 0; tt < tokens.length && all; tt++) {
          if (entry.lcHeadings[hh].lc.indexOf(tokens[tt]) < 0) all = false;
        }
        if (all) headingHit = entry.lcHeadings[hh];
      }

      best = headingHit
        ? { points: 34, where: headingHit, at: headingHit.at }
        : { points: 18, where: firstAt >= 0 ? nearestHeading(entry, firstAt) : null, at: firstAt };
    }

    if (!best) return null;

    // Short titles beat long ones on an equal match: "Provenance" is a better
    // answer to "provenance" than a page that merely discusses it at length.
    best.points += Math.max(0, 12 - entry.lcTitle.length * 0.1);
    return best;
  }

  /** The last heading starting at or before an offset, so a hit links to a section. */
  function nearestHeading(entry, offset) {
    var found = null;
    for (var i = 0; i < entry.lcHeadings.length; i++) {
      if (entry.lcHeadings[i].at <= offset) found = entry.lcHeadings[i];
      else break;
    }
    return found;
  }

  // ── Rendering ────────────────────────────────────────────────────────────

  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function highlight(text, query, tokens) {
    var needles = [query].concat(tokens).filter(function (n) { return n.length > 1; });
    // Longest first, so "ethical wall" wins over "wall" and the shorter needle
    // does not carve up the longer match.
    needles.sort(function (a, b) { return b.length - a.length; });

    var out = escapeHtml(text);
    var seen = {};
    for (var i = 0; i < needles.length; i++) {
      var n = needles[i].toLowerCase();
      if (seen[n]) continue;
      seen[n] = true;
      var re = new RegExp('(' + n.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
      // Skip anything already inside a <mark>, so nesting cannot happen.
      out = out.replace(/(<mark>[\s\S]*?<\/mark>)|([^<]+)/g, function (m, marked, plain) {
        if (marked) return marked;
        return plain.replace(re, '<mark>$1</mark>');
      });
    }
    return out;
  }

  function snippetFor(entry, hit, query, tokens) {
    // A title match has no position in the body, and falling back to a search would
    // find the page's own h1 — quoting the heading back at the reader. The lede
    // paragraph is what the page is about, which is what they actually want.
    if (hit.where === null && entry.summary) return entry.summary;

    var at = hit.at;
    if (at < 0) {
      var lcq = query.toLowerCase();
      at = entry.lcBody.indexOf(lcq);
      if (at < 0 && tokens.length) at = entry.lcBody.indexOf(tokens[0]);
    }
    if (at < 0) return entry.summary || entry.body.slice(0, 170);

    var start = Math.max(0, at - SNIPPET_RADIUS);
    var end = Math.min(entry.body.length, at + query.length + SNIPPET_RADIUS * 2);
    // Snap to word boundaries so a snippet does not open mid-word.
    if (start > 0) {
      var sp = entry.body.indexOf(' ', start);
      if (sp >= 0 && sp < at) start = sp + 1;
    }
    var text = entry.body.slice(start, end).trim();
    return (start > 0 ? '… ' : '') + text + (end < entry.body.length ? ' …' : '');
  }

  function search(query) {
    var tokens = tokenise(query);
    var results = [];
    for (var i = 0; i < entries.length; i++) {
      var hit = score(entries[i], query, tokens);
      if (hit) results.push({ entry: entries[i], hit: hit });
    }
    results.sort(function (a, b) { return b.hit.points - a.hit.points; });
    return results.slice(0, MAX_RESULTS);
  }

  function render(query) {
    var results = search(query);
    hits = [];
    activeIndex = -1;

    var html = '';
    html += '<div class="search-panel">';
    html += '<div class="search-panel-head"><span>' +
      (results.length ? results.length + (results.length === 1 ? ' result' : ' results') : 'No results') +
      ' for “' + escapeHtml(query) + '”</span>' +
      '<span><kbd>↑</kbd><kbd>↓</kbd> move &nbsp;<kbd>enter</kbd> open &nbsp;<kbd>esc</kbd> close</span></div>';

    if (!results.length) {
      html += '<div class="search-empty">Nothing matched. Try a single word — ' +
        '<a href="glossary.html">the glossary</a> covers every term the docs use.</div>';
    } else {
      // Grouped by page: one page contributing three sections should read as one
      // place to look, not three competing answers.
      var currentPath = null;
      for (var i = 0; i < results.length; i++) {
        var r = results[i];
        if (r.entry.path !== currentPath) {
          currentPath = r.entry.path;
          html += '<div class="search-group-title">' + escapeHtml(r.entry.title) + '</div>';
        }
        var href = r.entry.path + (r.hit.where ? '#' + r.hit.where.id : '');
        // The group heading already names the page, so a whole-page hit is labelled
        // by what it is rather than repeating the title on the line beneath it.
        var label = r.hit.where ? r.hit.where.text : 'Overview of this page';
        var snippet = snippetFor(r.entry, r.hit, query, tokenise(query));
        hits.push(href);
        html += '<a class="search-hit" href="' + href + '" data-i="' + (hits.length - 1) + '">';
        html += '<span class="hit-title">' + highlight(label, query, tokenise(query)) + '</span>';
        html += '<span class="hit-snippet">' + highlight(snippet, query, tokenise(query)) + '</span>';
        html += '</a>';
      }
    }
    html += '</div>';

    panel.innerHTML = html;
    overlay.classList.add('open');
    if (hits.length) setActive(0);
  }

  function setActive(i) {
    var nodes = panel.querySelectorAll('.search-hit');
    if (!nodes.length) return;
    if (i < 0) i = nodes.length - 1;
    if (i >= nodes.length) i = 0;
    for (var n = 0; n < nodes.length; n++) nodes[n].classList.remove('active');
    nodes[i].classList.add('active');
    activeIndex = i;
    if (nodes[i].scrollIntoView) nodes[i].scrollIntoView({ block: 'nearest' });
  }

  function close() {
    overlay.classList.remove('open');
    activeIndex = -1;
    hits = [];
  }

  // ── Wiring ───────────────────────────────────────────────────────────────

  function init() {
    box = document.getElementById('search-box');
    if (!box) return;

    overlay = document.createElement('div');
    overlay.className = 'search-results';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-label', 'Search results');
    panel = document.createElement('div');
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    box.addEventListener('input', function () {
      var q = box.value.trim();
      if (q.length < 2) { close(); return; }
      render(q);
    });

    box.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { close(); box.blur(); return; }
      if (!overlay.classList.contains('open')) return;
      if (e.key === 'ArrowDown') { e.preventDefault(); setActive(activeIndex + 1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(activeIndex - 1); }
      else if (e.key === 'Enter') {
        e.preventDefault();
        if (activeIndex >= 0 && hits[activeIndex]) window.location.href = hits[activeIndex];
      }
    });

    overlay.addEventListener('click', function (e) {
      // Clicking the backdrop closes; clicking a result follows the link.
      if (e.target === overlay) close();
    });

    document.addEventListener('keydown', function (e) {
      var tag = (e.target.tagName || '').toLowerCase();
      var typing = tag === 'input' || tag === 'textarea' || tag === 'select';
      if (e.key === '/' && !typing) { e.preventDefault(); box.focus(); box.select(); }
      else if (e.key === 'Escape' && overlay.classList.contains('open')) close();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
