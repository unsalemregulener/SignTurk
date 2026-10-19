# Changelog

Notable changes to SignTurk. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-19

The graduation release, named an Outstanding Graduation Project at Eastern
Mediterranean University (2025–2026).

### Recognition

- All 226 AUTSL signs, read from RTMW whole-body landmarks by a four-stream
  ensemble: 94.39% top-1 and 98.64% top-3 on the AUTSL test split.
- Four-second recordings; the sign is cut out where the hands rise and come
  back to rest.
- Hands-free mode on a GPU: every sign is cut out of the camera stream and
  waits for approval, and confident results can enter the sentence
  automatically.
- A framing check with a silhouette asks the signer to sit back or tilt the
  camera until the space below the shoulders is in view.

### Interface

- Home, translation, dictionary, history, settings and administration pages
  in light and dark themes, with a three-step camera setup on the first visit.
- A word enters the sentence only after approval; Enter approves, Esc rejects
  and Space records.
- A rule-based Turkish sentence engine, speech output and a 3D avatar that
  plays every sign.

### Platform

- Token sign-in with expiring tokens and limited login attempts, and per-user
  history and settings.
- SQLite with no setup, or PostgreSQL through `DATABASE_URL`; a Docker image;
  CI that runs the real model end to end.

[1.0.0]: https://github.com/unsalemregulener/SignTurk/releases/tag/v1.0.0
