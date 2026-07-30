import assert from "node:assert/strict";
import {
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import test from "node:test";
import ts from "typescript";

import { WORKBENCH_NAV_GROUPS } from "../src/workbenchRoutes.ts";

const sourceRoot = resolve(import.meta.dirname, "../src");
const pageShell = resolve(import.meta.dirname, "../index.html");
const appSource = readFileSync(
  resolve(import.meta.dirname, "../src/App.tsx"),
  "utf8",
);
const reconstructSource = readFileSync(
  resolve(import.meta.dirname, "../src/components/ReconstructWorkbench.tsx"),
  "utf8",
);
const exactSourcesWithHistoricalCommentDashes = new Set([
  resolve(import.meta.dirname, "../src/components/CubeState.tsx"),
  resolve(import.meta.dirname, "../src/lib/cubePerms.data.ts"),
]);

function sourceFiles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) return sourceFiles(path);
    return /\.(?:ts|tsx)$/.test(entry.name) ? [path] : [];
  });
}

function withoutBlockAndFullLineComments(source) {
  return source
    .replace(/\/\*[\s\S]*?\*\//gu, "")
    .replace(/^\s*\/\/.*$/gmu, "");
}

test("workbench copy contains no em dashes", () => {
  assert.doesNotMatch(readFileSync(pageShell, "utf8"), /—/u, pageShell);
  for (const path of sourceFiles(sourceRoot)) {
    const source = readFileSync(path, "utf8");
    const visibleSource = exactSourcesWithHistoricalCommentDashes.has(path)
      ? withoutBlockAndFullLineComments(source)
      : source;
    assert.doesNotMatch(visibleSource, /—/u, path);
  }
});

// A semicolon that joins two independent clauses is the tell that made the
// workbench copy read as machine-written, and the documentation drifted to 129
// of them before anyone counted. Only user-visible prose is checked: TypeScript
// statement separators, type-member separators, and `for` headers never reach
// this rule, because the check reads the parsed syntax tree and looks at string
// literals, template literal chunks, and JSX text nodes only.
//
// The signal is `;` followed by whitespace and a lowercase letter. That misses
// a semicolon before a capitalized word, which is the acceptable cost of never
// flagging code.
const PROSE_SEMICOLON = /;\s+\p{Ll}/u;

// HTML entities end in `;`, so `35&nbsp;mm` and `it&apos;s` would read as
// violations. Dropping the entity first cannot hide a real one, because the
// substitution removes a semicolon rather than adding one.
const HTML_ENTITY = /&[a-zA-Z][a-zA-Z0-9]*;/gu;

// A CSS declaration also ends in `;`, so a string holding style rules is code
// rather than copy. Only a bare `property: value;` run is dropped: the value
// may not contain sentence punctuation, which keeps a real prose semicolon
// later in the same sentence from being swallowed by the match.
const CSS_DECLARATION = /[-a-zA-Z]+\s*:\s*[^;:{}.!?]*;/gu;

// Exact string values that carry a semicolon for a non-prose reason. Keep this
// empty unless a real case appears, and add the exact literal rather than a
// pattern, so an addition stays a reviewed one-line decision.
const NON_PROSE_LITERALS = new Set([]);

function visibleStrings(path) {
  const file = ts.createSourceFile(
    path,
    readFileSync(path, "utf8"),
    ts.ScriptTarget.Latest,
    true,
    path.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  );
  const found = [];
  const visit = (node) => {
    if (
      ts.isStringLiteral(node) ||
      ts.isNoSubstitutionTemplateLiteral(node) ||
      ts.isTemplateHead(node) ||
      ts.isTemplateMiddle(node) ||
      ts.isTemplateTail(node) ||
      ts.isJsxText(node)
    ) {
      // Module specifiers are paths, never prose.
      const parent = node.parent;
      const isSpecifier =
        (ts.isImportDeclaration(parent) || ts.isExportDeclaration(parent)) &&
        parent.moduleSpecifier === node;
      if (!isSpecifier) {
        found.push({
          text: node.text,
          line:
            file.getLineAndCharacterOfPosition(node.getStart(file)).line + 1,
        });
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return found;
}

export function proseSemicolonViolations(paths) {
  const violations = [];
  for (const path of paths) {
    for (const { text, line } of visibleStrings(path)) {
      if (NON_PROSE_LITERALS.has(text)) continue;
      const copy = text
        .replace(HTML_ENTITY, " ")
        .replace(CSS_DECLARATION, " ");
      if (!PROSE_SEMICOLON.test(copy)) continue;
      violations.push(`${path}:${line} ${text.replace(/\s+/gu, " ").trim()}`);
    }
  }
  return violations;
}

test("workbench copy joins no independent clauses with a semicolon", () => {
  assert.deepEqual(proseSemicolonViolations(sourceFiles(sourceRoot)), []);
});

test("the semicolon rule reads prose and leaves TypeScript syntax alone", () => {
  // The fixture lives outside the repository so the first test, `tsc`, and Vite
  // never see it.
  const scratch = mkdtempSync(resolve(tmpdir(), "cubed-copy-style-"));
  const fixture = resolve(scratch, "fixture.tsx");
  const probe = [
    "interface Size { width: number; height: number }",
    'const style = { display: "flex" };',
    "const css = `.a { color: red; margin: 0 }`;",
    "function count(): number { let total = 0; for (let i = 0; i < 3; i += 1) total += i; return total; }",
    'const label = "Sealed. Keep it on your own disk.";',
    'const focal = <small>35&nbsp;mm-equivalent focal length</small>;',
    "export function Fixture() { return <p>Decode still works. The extras stay hidden.</p>; }",
  ].join("\n");

  try {
    writeFileSync(fixture, probe, "utf8");
    assert.deepEqual(
      proseSemicolonViolations([fixture]),
      [],
      "TypeScript syntax must never be flagged",
    );

    writeFileSync(
      fixture,
      probe.replace(
        "Decode still works. The extras stay hidden.",
        "Decode still works without this; its extras stay hidden.",
      ),
      "utf8",
    );
    const caught = proseSemicolonViolations([fixture]);
    assert.equal(caught.length, 1, "a reintroduced prose semicolon must fail");
    assert.match(caught[0], /without this; its extras stay hidden/);
  } finally {
    rmSync(scratch, { force: true, recursive: true });
  }
});

// The same rule applies to reader-facing Markdown. Markdown has no syntax tree
// here, so the non-prose parts are removed line by line before the rule runs:
// fenced code, table rows, inline code spans, link targets, and URLs are not
// copy. A line ending in `;`, `; and`, or `; or` is an enumerated item in legal
// or contract drafting, which is the one place the semicolon is correct.
const repoRoot = resolve(import.meta.dirname, "../../..");
const READER_FACING_ROOT_DOCS = [
  "README.md",
  "CONTRIBUTING.md",
  "SECURITY.md",
];
const MARKDOWN_NON_PROSE = [
  [/`[^`]*`/gu, " "], // inline code span
  [HTML_ENTITY, " "],
  [/\]\([^)]*\)/gu, "] "], // link and image target
  [/<[a-z][a-z0-9+.-]*:[^>]*>/giu, " "], // autolink
  [/\b[a-z][a-z0-9+.-]*:\/\/\S+/giu, " "], // bare URL
  [/;(?:\s+(?:and|or))?\s*$/u, ""], // enumerated item terminator
];

function markdownFiles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) return markdownFiles(path);
    return entry.name.endsWith(".md") ? [path] : [];
  });
}

export function markdownProseSemicolonViolations(paths) {
  const violations = [];
  for (const path of paths) {
    let fenceMark = null;
    readFileSync(path, "utf8")
      .split("\n")
      .forEach((line, index) => {
        const fence = line.match(/^\s*(```+|~~~+)/u);
        if (fence) {
          if (fenceMark === null) fenceMark = fence[1][0];
          else if (fenceMark === fence[1][0]) fenceMark = null;
          return;
        }
        if (fenceMark !== null) return;
        if (/^\s*\|/u.test(line)) return; // table row
        const copy = MARKDOWN_NON_PROSE.reduce(
          (text, [pattern, replacement]) => text.replace(pattern, replacement),
          line,
        );
        if (!PROSE_SEMICOLON.test(copy)) return;
        violations.push(`${path}:${index + 1} ${line.trim()}`);
      });
  }
  return violations;
}

test("reader-facing documentation joins no clauses with a semicolon", () => {
  assert.deepEqual(
    markdownProseSemicolonViolations([
      ...READER_FACING_ROOT_DOCS.map((name) => resolve(repoRoot, name)),
      ...markdownFiles(resolve(repoRoot, "docs")),
    ]),
    [],
  );
});

test("the Markdown rule leaves enumerations, tables, and code alone", () => {
  const scratch = mkdtempSync(resolve(tmpdir(), "cubed-copy-style-md-"));
  const fixture = resolve(scratch, "fixture.md");
  const probe = [
    "A contribution qualifies when it has:",
    "",
    "- an accepted terms version;",
    "- a reviewed rights record; and",
    "- a public license, or a recorded exception.",
    "",
    "| Asset | Why |",
    "| --- | --- |",
    "| `clip.mp4` | Source recording; not needed for the replay |",
    "",
    "```bash",
    "for f in *.mp4; do probe \"$f\"; done",
    "```",
    "",
    "Set `a: 1; b: 2` in the header. See [the policy](PRIVACY.md#a-note) first.",
    "The validator rejects them. They must not appear in public bundles.",
  ].join("\n");

  try {
    writeFileSync(fixture, probe, "utf8");
    assert.deepEqual(
      markdownProseSemicolonViolations([fixture]),
      [],
      "correct legal, table, and code semicolons must never be flagged",
    );

    writeFileSync(
      fixture,
      probe.replace(
        "The validator rejects them. They must not appear in public bundles.",
        "The validator rejects them; they must not appear in public bundles.",
      ),
      "utf8",
    );
    const caught = markdownProseSemicolonViolations([fixture]);
    assert.equal(caught.length, 1, "a reintroduced prose semicolon must fail");
    assert.match(caught[0], /rejects them; they must not appear/);
  } finally {
    rmSync(scratch, { force: true, recursive: true });
  }
});

test("navigation keeps its exact grouped structure", () => {
  assert.deepEqual(
    WORKBENCH_NAV_GROUPS.map(({ label, items }) => ({
      label,
      items: items.map(({ label: item, to, badge }) => ({ item, to, badge })),
    })),
    [
      {
        label: null,
        items: [
          { item: "Demo", to: "/demo", badge: undefined },
          { item: "Decode", to: "/decode", badge: undefined },
          { item: "Runs", to: "/runs", badge: undefined },
        ],
      },
      {
        label: "Data tools",
        items: [
          { item: "Add video", to: "/import", badge: undefined },
          { item: "Label", to: "/label", badge: undefined },
        ],
      },
    ],
  );
  assert.doesNotMatch(appSource, /className="nav-status/);
  assert.doesNotMatch(
    appSource,
    /capability-badge capability-ready">Available/,
  );
  assert.doesNotMatch(reconstructSource, /AnalysisWorkbench/);
});
