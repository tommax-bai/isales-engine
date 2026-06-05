"""Streaming helpers for the main LLM → TTS path (pipeline-stream-and-referee).

- ``sentence_splitter`` slices a token stream into TTS-ready sentences.
- ``types`` holds the shared streaming value objects (RefereeResult).
"""
