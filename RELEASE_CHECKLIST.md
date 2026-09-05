# Public-release checklist

- [x] Export from a fixed source commit without private Git history.
- [x] Exclude caches, raw outputs, datasets, weights, credentials, and archives.
- [x] Record authorship, license, and source provenance.
- [x] Align public CLI defaults with the camera-ready configuration.
- [x] Add static CI, paper-default regression tests, and asset preflight.
- [x] Record the vLLM migration boundary and reject incompatible patching.
- [ ] Confirm the MultiChallenge summary-generation artifact/procedure.
- [ ] Confirm the redistributable preparation path for MathVista captions/OCR.
- [ ] Run the GPU smoke test with vLLM 0.28.0 after clean installation.
- [x] Run static secret-pattern and dependency scans on the candidate.
- [ ] Resolve or formally accept advisories caused by the vLLM 0.28.0
  compatibility pin; keep the repository private until this is reviewed.
- [ ] Obtain collaborator/compliance approval for the exact release commit.
- [ ] Change repository visibility from private to public.
