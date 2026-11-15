# Pyodide 314.0.8 (vendored)

CPython 3.14 compiled to WebAssembly. The workbench runs the `parallelogram`
package on it, in the visitor's tab. These five files are copied unchanged
from the npm package `pyodide@314.0.8` (https://pyodide.org, MPL-2.0; the
bundled Python standard library is under the PSF licence):

    pyodide.mjs  pyodide.asm.mjs  pyodide.asm.wasm  python_stdlib.zip  pyodide-lock.json

They are served from this site rather than a CDN so that the sealed route can
keep `connect-src 'self'`. To verify them against the published package:

    npm pack pyodide@314.0.8 && tar -xzf pyodide-314.0.8.tgz
    (cd package && sha256sum -c ../SHA256SUMS)

To upgrade: add a new `pyodide-<version>/` folder, point `engine-worker.js` at
it, and delete this one. The folder name carries the version because these
files are cached for a year (`landing/vercel.json`).

No Pyodide package beyond the standard library is loaded: the engine's check
and fix paths are pure standard-library Python.
