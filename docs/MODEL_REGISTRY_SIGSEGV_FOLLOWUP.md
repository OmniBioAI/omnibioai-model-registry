# Model registry SIGSEGV — follow-up (re-assessment + cleanup)

Follow-up to `MODEL_REGISTRY_SIGSEGV_RESOLUTION.md` (workbench repo,
`obsolete/`). That report diagnosed a real SIGSEGV in a Cython-compiled
`read_sha256_manifest`/`verify_sha256_manifest` (`manifest.py`, compiled with
`boundscheck=False`/`wraparound=False` for the in-progress "IP protection"
feature, `234e372` onward), confirmed the real pinned `v0.1.1` Docker images
are unaffected, and recommended — but did not perform — deleting the
crash-causing `.so` files from this dev host. This follow-up: (1)
re-checks whether `validate.py` drifted into the same risk since that
report, (2) actually executes the recommended `.so` cleanup and verifies it,
(3) makes the case for escalating to the feature owner.

## 1. Did `validate.py` change near `read_sha256`? — No, and it never has

The task that generated this document was told `validate.py` "has since
changed near `read_sha256`." That premise does not hold up:

- `read_sha256_manifest`/`verify_sha256_manifest` have **never lived in
  `validate.py`** — they're in `manifest.py`, and always have been.
  `validate.py` contains exactly one function, `validate_package_files()`,
  which does a list-comprehension existence check
  (`(version_dir / f).exists()`) — no manifest parsing, no string/byte
  indexing, none of the `GetItemInt_Fast` machinery that crashed.
- `validate.py`'s **entire history**, from its first commit
  (`fba0944`, "Initiate model registry") to `HEAD` (`2ab1924`, current
  `main`), is 4 commits, and the only changes across all of them are
  cosmetic: blank-line/import-order formatting (`3e3ddfb` "apply black
  formatting", `21f7e2e` "auto-fix ruff I001 import sorting"). No logic was
  ever added, removed, or touched.
- Diffed against the `v0.1.1` tag directly: same result — `manifest.py`
  (+11/-9 lines) and `validate.py` (+8/-6 lines) differ only in whitespace
  and import ordering, not logic. (Correction to the original report,
  which called these two files "byte-for-byte identical" between `v0.1.1`
  and `HEAD` — they are not byte-identical, just logically unchanged; the
  substance of that report's conclusion still holds.)
- The one file that *did* gain real content since `v0.1.1` is `localfs.py`
  — a new `write_once_text()` method (atomic exclusive-create via
  `os.link()`). It's unrelated to the crash site and not in the crash's
  call path (`resolve_model()` → `validate_package_files()` →
  `verify_sha256_manifest()` → `read_sha256_manifest()`); the original
  report never actually checked `localfs.py`'s diff against `v0.1.1`
  either (it only names `validate.py`/`manifest.py`), so this is worth
  knowing but isn't new risk today.

**Conclusion: nothing "near `read_sha256`" changed in `validate.py`, because
that logic was never there. No recompile-and-crash-test was warranted for
`validate.py` under the report's own "if it touches the same logic"
condition — it doesn't.** Instead, effort went into re-confirming the real
crash site directly (§2), since that's where actual risk lives.

## 2. Live re-confirmation of the crash (before touching anything)

Before any cleanup, reproduced the original repro fresh, right now, on this
host — same command shape as the original report
(`resolve_model("ptm_site_prediction", "ptm_head_real_v1@production",
verify=True)` against the real registry root):

```
$ python3 repro.py
Segmentation fault (core dumped)     # exit 139
```

`gdb` backtrace, captured live:

```
Program received signal SIGSEGV, Segmentation fault.
0x0000fffff6386c44 in __Pyx_GetItemInt_Fast.constprop.0 ()
   from .../omnibioai_model_registry/package/manifest.cpython-313-aarch64-linux-gnu.so
#0  __Pyx_GetItemInt_Fast.constprop.0 () from .../manifest.cpython-313-aarch64-linux-gnu.so
#1  __pyx_pw_24omnibioai_model_registry_7package_8manifest_5read_sha256_manifest ()
#2  __pyx_pw_24omnibioai_model_registry_7package_8manifest_7verify_sha256_manifest ()
#3  _PyObject_VectorcallTstate (... CPython 3.13.9 ...)
```

**Identical crash site to the original report** — same function, same
`.so`, same `GetItemInt_Fast` indexing helper. The `.so` files were still
present with the exact mtimes the original report cited
(`manifest.cpython-313-aarch64-linux-gnu.so` et al., `Aug 26 15:04`) —
confirming this is the same stray build artifact, not a new build, sitting
unremoved for several days after diagnosis.

## 3. Cleanup — executed and verified

Deleted the three `.so` files named in the original report's recommended
cleanup command, plus the `build/` directory (same build byproduct,
already gitignored, not previously removed):

```
$ rm omnibioai_model_registry/package/manifest.cpython-313-aarch64-linux-gnu.so \
     omnibioai_model_registry/package/validate.cpython-313-aarch64-linux-gnu.so \
     omnibioai_model_registry/storage/localfs.cpython-313-aarch64-linux-gnu.so
$ rm -rf build/
```

Re-ran the exact same real repro afterward:

```
now loading from: .../omnibioai_model_registry/package/manifest.py   # falls back to source, as expected
$ python3 repro.py
OK -> .../tasks/ptm_site_prediction/models/ptm_head_real_v1/versions/2026-08-08_010032
```
Exit code 0. Also re-ran the equivalent real `resolve_model(..., verify=True)`
call for the other two plugins from the original report, for full parity:

```
microbiome_taxonomy -> OK -> .../taxa_cnn_real_v1/versions/2026-08-08_005445
amr_gene_classification -> OK -> .../amr_cnn_real_v1/versions/2026-08-08_081410
```

**Bare-metal usage of this package on this host no longer crashes.**
`.so`/`build/` were already gitignored, so this cleanup produced no `git
status` change — verified (`git status --porcelain` before and after is
identical, still only the three known `.c` regeneration diffs — untouched,
per instruction, pending the escalation below).

## 4. This needs to go to the feature owner — it's live on `main`, not contained

This is not a self-contained local experiment:

- `234e372` ("feat: add Cython IP protection - 434 compiled .so binaries")
  **is merged into `main`** (`git merge-base --is-ancestor 234e372
  2ab1924` → yes). `main` is also the branch currently checked out — there
  is no separate feature branch isolating this work. Anyone who clones
  `main` and runs `python setup.py build_ext --inplace` (the documented
  usage in `setup.py`'s own docstring) gets the same
  `boundscheck=False`/`wraparound=False` compilation of `manifest.py`,
  `validate.py`, `localfs.py` — and, per this report, the same crash, on at
  least this CPython 3.13.9 build.
- The tracking policy is inconsistent and confusing on its own, independent
  of the crash: the generated `.c` intermediates (`manifest.c`,
  `validate.c`, `localfs.c` — ~30K lines total) **are committed to git**,
  while the compiled `.so` outputs from the same build are gitignored. This
  means every local rebuild with a different Cython version (confirmed:
  `HEAD`'s committed `manifest.c` says `Generated by Cython 3.2.9`; this
  host's regenerated one says `3.3.0`) produces large, meaningless diffs on
  the tracked `.c` files — the same class of noise as the `.egg-info`/
  `coverage/` cleanups done elsewhere, except here the file in question is
  also the literal crash site, so "just discard the diff" isn't a safe
  reflex without someone confirming intent first.
- Recommendation: flag directly to whoever owns the Cython IP-protection
  work, before it's built or released anywhere else:
  1. `read_sha256_manifest` segfaults when Cython-compiled with
     `boundscheck=False`/`wraparound=False` under CPython 3.13.9 — real,
     reproducible, gdb-confirmed twice now (original report + this one).
  2. The feature is already on `main`, not isolated to a branch — anyone
     building it hits this today, not just this one dev host.
  3. Decide and fix the tracking policy for generated `.c`/`.so` output
     (most likely: stop committing the `.c` files at all, same as `.so`,
     and regenerate both at build time) — separately from the crash fix,
     since it's what's producing the noisy diffs currently sitting
     uncommitted in this checkout.
  4. Until the Cython path is fixed, the `.py` source remains the safe
     fallback (confirmed unaffected, §1) — anyone hitting this can delete
     the local `.so` (§3) as an immediate workaround.

## What's still open / not done here

- The three `.c` files (`manifest.c`, `validate.c`, `localfs.c`) remain
  modified in the working tree, uncommitted and undiscarded, per
  instruction — they're pure Cython-version regeneration noise (§1) but are
  left in place in case they're useful reference while this gets
  escalated.
- No fix to the Cython compiler directives, no commit, no push, no PR was
  made — this is a diagnosis + cleanup + escalation recommendation only.
