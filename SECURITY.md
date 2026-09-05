# Security policy

## Research-artifact scope

This release reproduces the paper implementation by patching vLLM 0.28.0.
That compatibility pin and several transitive dependencies have known
security advisories. The code should therefore be treated as an offline
research artifact, not as a hardened production inference server.

Until the patches are ported and validated against a supported vLLM release:

- run them in a dedicated, least-privilege environment;
- use only trusted model weights, datasets, prompts, and serialized artifacts;
- do not expose the patched runtime directly to untrusted network clients;
- keep credentials outside the repository and rotate any accidentally exposed
  token immediately;
- review generated `.pt` vocabulary maps before loading them, since PyTorch
  serialization is not a safe interchange format for untrusted files.

The release-candidate dependency audit was performed on 2026-08-30. Public
release requires either a validated compatibility update or explicit risk
acceptance by the project maintainers.

## Reporting

Please report suspected vulnerabilities privately to
`lvhang1001@mail.ustc.edu.cn` before opening a public issue.
