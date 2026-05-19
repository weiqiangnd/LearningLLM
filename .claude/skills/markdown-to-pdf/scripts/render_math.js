#!/usr/bin/env node
// Read JSON-per-line from stdin, render each TeX snippet via KaTeX,
// emit the rendered HTML strings one-per-line on stdout (JSON-encoded).
// Input:  {"tex":"<latex>", "display": true|false}
// Output: <json-string-of-html>   (or {"error":"..."} as JSON object on failure)

const path = require("path");
const readline = require("readline");

const skillDir = path.resolve(__dirname, "..");
const katex = require(path.join(skillDir, "node_modules", "katex"));

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

rl.on("line", (line) => {
  if (!line) {
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
    // `htmlAndMathml` makes KaTeX emit both the visual HTML span and a MathML
    // `<annotation encoding="application/x-tex">SOURCE</annotation>` carrying
    // the exact TeX input. The downstream verification step reads those
    // annotations back out of the final HTML and checks them against the math
    // we fed in — that's the roundtrip anchor for detecting silent rendering
    // anomalies (lost subscripts, swallowed punctuation, etc.).
    const html = katex.renderToString(item.tex, {
      displayMode: !!item.display,
      throwOnError: false,
      strict: "ignore",
      output: "htmlAndMathml",
      trust: false,
    });
    process.stdout.write(JSON.stringify(html) + "\n");
  } catch (e) {
    process.stdout.write(JSON.stringify({ error: e.message }) + "\n");
  }
});
