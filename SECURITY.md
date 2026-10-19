# Security policy

## Reporting a vulnerability

Please report security problems privately: open the repository's
**Security** tab and choose **Report a vulnerability**. Do not open a public
issue for them.

Include the steps to reproduce, the affected route or file, and what an
attacker could do with it.

## Supported versions

Fixes go into the latest release on `main`; older releases are not patched.

## Scope

In scope: the FastAPI backend (accounts, tokens, login limits, the live
WebSocket), the web interface and the Docker image. The model weights and the
AUTSL-derived data are out of scope.
