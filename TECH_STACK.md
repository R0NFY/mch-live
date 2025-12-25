# Tech Stack

- **Core Framework**: [LiveKit Agents](https://docs.livekit.io/agents/) (Python SDK)
- **Real-time Communication**: [LiveKit Cloud](https://livekit.io/)
- **Large Language Model (LLM)**: Qwen (via LiveKit inference)
- **Speech-to-Text (STT)**: Cartesia/Ink-Whisper
- **Text-to-Speech (TTS)**: Inworld TTS
- **Avatar Integration**: [Simli](https://www.simli.com/)
- **Package Manager**: `uv`
- **Environment Management**: `python-dotenv`
- **Audio Processing**: Silero VAD, Noise Cancellation (BVC)
- **Infrastructure (Proposed)**: [Yandex Cloud](https://cloud.yandex.ru/)
    - **Object Storage**: S3-compatible storage for media.
    - **Cloud Functions**: Serverless endpoint for transcript processing.
