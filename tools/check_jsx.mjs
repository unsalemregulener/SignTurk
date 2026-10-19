// Compiles the interface's JSX with the Babel build the page loads, so a syntax
// error fails here instead of in a browser.
//
//   npm install --no-save @babel/standalone@<version in frontend/app/index.html>
//   node tools/check_jsx.mjs [file ...]   (default: every frontend/app/js/**/*.jsx)
import { readdirSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const Babel = require("@babel/standalone");
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function walk(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return walk(path);
    return path.endsWith(".jsx") ? [path] : [];
  });
}

const files = process.argv.length > 2
  ? process.argv.slice(2).map((path) => resolve(path))
  : walk(join(root, "frontend", "app", "js")).sort();

let failed = 0;
for (const file of files) {
  try {
    Babel.transform(readFileSync(file, "utf8"), { presets: ["react"], filename: file });
    console.log("ok  ", relative(root, file));
  } catch (error) {
    failed += 1;
    console.log("FAIL", relative(root, file), String(error.message).split("\n")[0]);
  }
}
console.log(`Babel ${Babel.version}: ${files.length} file(s), ${failed} failed`);
process.exit(failed ? 1 : 0);
