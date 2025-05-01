This document describes the architectural plan, services, and components for use in code generation and agent grounding.

## System Diagram

```
[WebApp] → [Backend] → [DeepGram (STT)] → [RAG Engine] → [Pinecone + LLM] → [DeepGram (TTS)] → [WebApp]
```

## Stack

- Frontend: React (Swift UI), DeepGram Web SDK
- Backend: Python (FastAPI)
- Vector DB: Pinecone
- LLM: OpenAI (GPT-4)
- Voice: DeepGram STT/TTS
- Auth: Firebase/Auth0
- DB: PostgreSQL (Supabase or AWS RDS)

## Core Services

### /api/call/start

Starts a new session, returns a session ID.

### /api/call/transcript

Receives transcription from DeepGram, forwards to RAG engine.

### /api/rag/query

Performs Pinecone search → LLM response → returns text.

### /api/tts/speak

Returns DeepGram TTS stream.

### /api/analytics/report

Returns JSON summary of past sessions.

### /api/admin/export

Returns downloadable CSV of analytics.
