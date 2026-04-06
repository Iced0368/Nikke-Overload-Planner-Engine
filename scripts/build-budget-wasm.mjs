import { spawnSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

function resolveEmcc() {
  if (process.env.EMCC) {
    return process.env.EMCC;
  }

  const locator = process.platform === "win32" ? ["where", ["emcc"]] : ["which", ["emcc"]];
  const located = spawnSync(locator[0], locator[1], { encoding: "utf8" });
  if (located.status === 0) {
    const firstLine = located.stdout.split(/\r?\n/).find((line) => line.trim().length > 0);
    if (firstLine) {
      return firstLine.trim();
    }
  }

  const bashLocated = spawnSync("bash", ["-lc", "command -v emcc"], { encoding: "utf8" });
  if (bashLocated.status === 0) {
    const bashPath = bashLocated.stdout.trim();
    if (bashPath) {
      return bashPath;
    }
  }

  throw new Error("emcc를 찾을 수 없습니다. EMCC 환경 변수를 지정하거나 Emscripten을 PATH에 추가하세요.");
}

function shellQuote(value) {
  return `'${value.replace(/'/g, `"'"'`)}'`;
}

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const inputPath = resolve(projectRoot, "wasm", "budgetOptimizerCore.cpp");
const outputPath = resolve(projectRoot, "public", "wasm", "budgetOptimizerCore.mjs");
const emcc = resolveEmcc();
const emccArgs = [
  inputPath,
  "-O3",
  "-std=c++17",
  "-s",
  "WASM=1",
  "-s",
  "MODULARIZE=1",
  "-s",
  "EXPORT_ES6=1",
  "-s",
  "ENVIRONMENT=web,worker,node",
  "-s",
  "FILESYSTEM=0",
  "-s",
  "ALLOW_MEMORY_GROWTH=1",
  "-s",
  "EXPORT_ALL=1",
  "-s",
  'EXPORTED_FUNCTIONS=["_malloc","_free","_run_budget_optimizer_core"]',
  "-o",
  outputPath,
];

mkdirSync(dirname(outputPath), { recursive: true });

const result =
  process.platform === "win32"
    ? spawnSync("bash", ["-lc", [shellQuote(emcc), ...emccArgs.map(shellQuote)].join(" ")], {
        stdio: "inherit",
      })
    : spawnSync(emcc, emccArgs, {
        stdio: "inherit",
      });

if (result.status !== 0) {
  throw new Error(`emcc 빌드가 실패했습니다. 종료 코드: ${result.status ?? -1}`);
}

console.log(`Built budget optimizer WASM module: ${outputPath}`);
