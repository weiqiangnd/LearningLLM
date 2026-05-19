#!/usr/bin/env node
// Read JSON-per-line from stdin, render each TeX snippet via MathJax 3,
// emit one JSON per input on stdout. Mirrors the protocol used by the
// markdown-to-pdf skill's render_math.js but uses MathJax instead of KaTeX
// so we can verify the *GitHub* rendering path rather than our own.
//
// Input:   {"tex": "<latex>", "display": true|false}
// Output:  {"ok": true}               on successful render
//          {"error": "<message>"}     on parse error reported by MathJax

const path = require("path");
const readline = require("readline");

const skillDir = path.resolve(__dirname, "..");
const mjxRoot = path.join(skillDir, "node_modules", "mathjax-full", "js");

const { mathjax } = require(path.join(mjxRoot, "mathjax.js"));
const { TeX } = require(path.join(mjxRoot, "input", "tex.js"));
const { SVG } = require(path.join(mjxRoot, "output", "svg.js"));
const { liteAdaptor } = require(path.join(mjxRoot, "adaptors", "liteAdaptor.js"));
const { RegisterHTMLHandler } = require(path.join(mjxRoot, "handlers", "html.js"));
const { AllPackages } = require(path.join(mjxRoot, "input", "tex", "AllPackages.js"));

// liteAdaptor + RegisterHTMLHandler is the standard MathJax-3 server-side
// setup. AllPackages enables every TeX extension MathJax ships, which is
// (approximately) what github.com loads — github does NOT use a minimal
// MathJax bundle.
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);

const tex = new TeX({ packages: AllPackages });
const svg = new SVG();
const doc = mathjax.document("", { InputJax: tex, OutputJax: svg });

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

// MathJax 3 reports parse errors as a `<merror>` MathML node inside the SVG
// container. The error message is on the `data-mjx-error` attribute of the
// surrounding `<mjx-merror>` wrapper. If we see either marker, treat it as
// a failure and pull out the message for the Python side to log.
const ERROR_ATTR_RE = /data-mjx-error="([^"]*)"/;

rl.on("line", (line) => {
  if (!line.trim()) {
    process.stdout.write("\n");
    return;
  }
  let item;
  try {
    item = JSON.parse(line);
  } catch (e) {
    process.stdout.write(JSON.stringify({ error: "bad-json: " + e.message }) + "\n");
    return;
  }
  try {
    const node = doc.convert(item.tex, { display: !!item.display });
    const html = adaptor.outerHTML(node);
    if (html.includes("merror") || html.includes("data-mjx-error")) {
      const m = html.match(ERROR_ATTR_RE);
      const msg = m ? m[1] : "MathJax parse error (no detail)";
      // Still return `html` so the visual contact sheet can show the
      // red-flagged glyph alongside the source — useful when the error
      // message is terse and the rendered remnant clarifies the spot.
      process.stdout.write(JSON.stringify({ error: msg, html }) + "\n");
    } else {
      // `html` is the full `<mjx-container>...<svg>...</svg></mjx-container>`.
      // Visual mode inlines it directly into a contact-sheet HTML so the
      // SVG is rendered verbatim — no font dependencies, glyphs as paths.
      process.stdout.write(JSON.stringify({ ok: true, html }) + "\n");
    }
  } catch (e) {
    process.stdout.write(JSON.stringify({ error: e.message }) + "\n");
  }
});
