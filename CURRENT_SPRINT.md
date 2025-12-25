# Current Sprint: Post-Call Data Export

## Task List
- [x] Research LiveKit Egress for audio/video recording.
- [x] Research transcript export in LiveKit Agents.
- [x] Implement `on_session_finished` hook for transcript export.
- [x] Configure `LiveKitAPI` to trigger Egress for audio/video track recording.
- [x] Implement HTTP client to send data to the backend server.
- [ ] Add local amd64 smoke test for `analysis-worker` (OceanAI + GPT) via docker + S3.

## Status
- Currently investigating the best approach to capture and send:
    - Transcripts (AI & Person)
    - Audio (Person)
    - Video (Person)
- Currently validating `analysis-worker` behavior under `linux/amd64` locally to match production architecture.
