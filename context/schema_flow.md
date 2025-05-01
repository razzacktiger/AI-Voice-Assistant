=== Retrieval-Augmented Generation (RAG) Flow

The voice agent follows this flow to generate answers:

1. User initiates a voice call.
2. DeepGram transcribes voice input in real-time.
3. The backend retrieves the top-5 most relevant documents from Pinecone using the transcribed query.
4. Context documents + original query are passed to an LLM (OpenAI GPT-4) via a system prompt.
5. The LLM generates a conversational response.
6. Response is sent to DeepGram for Text-To-Speech (TTS) synthesis.
7. Audio reply is streamed back to the user.

=== Memory Store Schema

Conversation interactions are stored temporarily to allow context carry-over and future evaluation. Conversations older than 30 days are automatically deleted.

.Database: PostgreSQL
.Table: conversation_memory

| Field          | Type      | Description               |
| -------------- | --------- | ------------------------- |
| id             | SERIAL    | Primary Key               |
| session_id     | UUID      | Unique per call           |
| user_id        | UUID      | User making the call      |
| timestamp      | TIMESTAMP | When interaction happened |
| user_message   | TEXT      | Transcribed user speech   |
| agent_response | TEXT      | AI generated response     |

=== Analytics Service Schema

The analytics engine captures key performance indicators per call session. Export functionality is available via CSV generation on demand.

.Database: PostgreSQL
.Table: call_analytics

| Field              | Type      | Description                            |
| ------------------ | --------- | -------------------------------------- |
| id                 | SERIAL    | Primary Key                            |
| session_id         | UUID      | Foreign key to conversation            |
| user_id            | UUID      | User making the call                   |
| start_time         | TIMESTAMP | Call start time                        |
| end_time           | TIMESTAMP | Call end time                          |
| duration_seconds   | INT       | Call duration in seconds               |
| fallback_triggered | BOOLEAN   | True if AI could not answer a question |
| top_questions      | JSONB     | NLP summarized common queries per call |
