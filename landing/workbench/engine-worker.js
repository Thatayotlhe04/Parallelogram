// The engine, in a worker: Pyodide plus the parallelogram package from engine.zip.
// Every file it loads comes from this site. It never opens a connection anywhere else,
// and the Content-Security-Policy it is served with does not let it.
import { loadPyodide } from "./vendor/pyodide-314.0.8/pyodide.mjs";

const here = (path) => new URL(path, import.meta.url).href;
let py = null;
let bridge = null;

const post = (message, transfer) => self.postMessage(message, transfer || []);

async function boot() {
  const t0 = performance.now();
  post({ type: "boot", step: "runtime" });
  py = await loadPyodide({ indexURL: here("./vendor/pyodide-314.0.8/") });
  post({ type: "boot", step: "engine" });
  const archive = await (await fetch(here("./engine/engine.zip"))).arrayBuffer();
  py.FS.mkdirTree("/engine");
  py.FS.mkdirTree("/io");
  py.unpackArchive(archive, "zip", { extractDir: "/engine" });
  py.runPython("import sys\nsys.path.insert(0, '/engine')");
  bridge = py.pyimport("bridge");
  const version = py.runPython("import parallelogram, platform\nparallelogram.__version__ + '|' + platform.python_version()");
  const [engine, python] = version.split("|");
  post({ type: "ready", engine, python, pyodide: py.version, ms: Math.round(performance.now() - t0) });
}

const booted = boot().catch((error) => {
  post({ type: "fatal", message: String(error && error.message ? error.message : error) });
  throw error;
});

self.onmessage = async (event) => {
  const { id, name, args = {}, files = [] } = event.data;
  try {
    await booted;
    for (const file of files) py.FS.writeFile(file.path, new Uint8Array(file.bytes));
    const t0 = performance.now();
    const result = bridge.call(name, JSON.stringify(args));
    const ms = Math.round(performance.now() - t0);
    if (name === "export") {
      const bytes = py.FS.readFile(args.out_path);
      py.FS.unlink(args.out_path);
      post({ id, ok: true, ms, bytes: bytes.buffer }, [bytes.buffer]);
    } else {
      post({ id, ok: true, ms, result });
    }
    for (const file of files) py.FS.unlink(file.path);
  } catch (error) {
    // A Python exception arrives as a traceback; the last line is the message.
    const text = String(error && error.message ? error.message : error).trim();
    post({ id, ok: false, error: text.split("\n").pop(), trace: text });
  }
};
