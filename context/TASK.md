# AI Voice Assistant - Task List (Based on REQUIREMENTS.md & architecture.md)

# Last Updated: 2025-05-04

## Phase 1: Core Backend & API Setup

- [x] **Project Setup:** Create basic FastAPI app instance in `AI-Voice-Assistant-/api.py`. (Completed: 2025-05-04)
- [x] **Project Setup:** Create `requirements.txt` in project root. (Completed: 2025-05-04)
- [x] **Project Setup:** Add initial dependencies to `requirements.txt`. (Completed: 2025-05-04)
- [x] **Project Setup:** Install initial dependencies (`pip install -r requirements.txt`). (Completed: 2025-05-04)
- [x] **Project Setup:** Set up `.env` file structure and add initial keys (User confirmed). (Completed: 2025-05-04)
- [x] **Project Setup:** Create Python virtual environment (`venv`). (Completed: 2025-05-04)
- [x] **API Endpoint Stubs:** Create stub functions/routes in `api.py` for required endpoints. (Completed: 2025-05-04)
- [x] **Project Setup:** Verify `.gitignore` includes `venv/`, `__pycache__/`, `.env*`. (Completed: 2025-05-04)
- [ ] **Authentication (Firebase):**
  - [x] Initialize Firebase Admin SDK in `api.py`. (Completed: 2025-05-04)
  - [x] Implement basic user signup endpoint (e.g., `/auth/signup`). _(Handled by `get_current_user` upon first authenticated API call)_
  - [x] Implement basic user login endpoint (e.g., `/auth/login`). _(Authentication handled by token verification in `get_current_user`)_
  - [x] Add FastAPI dependency/middleware for verifying Firebase ID tokens. (Completed: 2025-05-04)
- [ ] **Database Setup (PostgreSQL):**
  - [x] Set up DB connection logic (using environment variables). (Completed: 2025-05-04)
  - [x] Define initial DB models (e.g., User, CallSession) using SQLModel. (Completed: 2025-05-04)
  - [x] Implement basic DB interaction for user creation/retrieval. _(Handled by `get_current_user`)_

## Phase 2: Real-time Voice Handling

- [x] **WebSocket Endpoint:**
  - [x] Implement WebSocket endpoint (`/ws`) in `api.py`. (Completed: 2025-05-04)
  - [x] Handle WebSocket connection/disconnection. (Completed: 2025-05-04)
  - [x] Add authentication check for WebSocket connections (using Firebase token). (Completed: 2025-05-04)
- [x] **Deepgram STT Integration:** (Partially Complete)
  - [x] Establish connection to Deepgram Streaming STT endpoint from the backend WebSocket handler. (Completed: 2025-05-04)
  - [x] Forward audio chunks received from the client WebSocket to Deepgram STT. (Completed & Tested via Simulation: 2025-05-04)
  - [x] Receive transcription results from Deepgram STT. (Completed: 2025-05-04 - Currently logs transcript; TODO: Integrate with RAG)

## Phase 3: RAG & LLM Integration

- [ ] **Pinecone Setup:**
  - [x] Implement Pinecone client initialization. (Completed: 2025-05-04)
  - [x] Implement function to query Pinecone index using OpenAI embeddings. (Completed: 2025-05-04)
- [x] **OpenAI LLM Call:**
  - [x] Implement OpenAI client initialization. (Completed: 2025-05-04)
  - [x] Create function to format prompt (transcript + Pinecone results) and call GPT-4. (Completed: 2025-05-04)
- [ ] **RAG Query Logic (`/api/rag/query` or WebSocket flow):**
  - [x] Integrate transcript reception -> Pinecone query -> LLM call -> response generation. (Completed: 2025-05-04 - Currently logs LLM response in handler)
  - [x] Implement basic session memory (e.g., store conversation history in DB or cache linked to session ID). (Completed: 2025-05-04 - Stores transcript/response in CallSession.conversation_history)

## Phase 4: TTS & Response Delivery

- [x] **Deepgram TTS Integration:** (Completed: 2025-05-05)
  - [x] Implement function to call Deepgram TTS with LLM response text (`api/tts.py`). (Completed: 2025-05-05)
  - [x] Stream TTS audio received from Deepgram back to the client via the WebSocket. (Completed: 2025-05-05)
- [x] **Connect RAG to TTS:** (Completed: 2025-05-05)
  - [x] Trigger TTS generation after receiving the final LLM response (`api/websocket.py`). (Completed: 2025-05-05)

## Phase 5: Logging, Analytics & Admin

- [ ] **Logging:**
  - [ ] Implement logging for calls, queries, and errors to the PostgreSQL database.
  - [ ] Implement 30-day rolling log retention logic.
- [ ] **Analytics Endpoint (`/api/analytics/report`):**
  - [ ] Implement logic to query logs and generate summary statistics. _(Stub exists)_
- [ ] **Admin Export (`/api/admin/export`):**
  - [ ] Implement logic to query analytics data and return as CSV. _(Stub exists)_

## Phase 6: Frontend & Deployment (Aligns with Original Week 1 & 2 Goals)

- [ ] **Frontend (React):**
  - [ ] Scaffold React frontend with login and call UI.
  - [ ] Connect frontend call button to backend WebSocket.
  - [ ] Create `/admin` dashboard route.
  - [ ] Display analytics on admin dashboard.
  - [ ] Add CSV export button functionality.
- [ ] **Deployment:**
  - [ ] Deploy frontend (e.g., Vercel).
  - [ ] Deploy backend (e.g., Railway/Render/AWS).

## Phase 7: Testing & Refinement

- [ ] **Unit Tests:**
  - [x] Set up `/tests` directory. (Completed: 2025-05-05)
  - [x] Write unit tests for authentication, API endpoints, RAG logic, etc. (Completed & Passed for refactored modules: 2025-05-05)
  - [x] Refactor tests into separate files per module. (Completed: 2025-05-05)
  - [x] Debug and fix failing tests after refactoring. (Completed: 2025-05-05)
- [ ] **Non-Functional Requirements:**
  - [ ] Test latency (< 1.5s).
  - [ ] Test concurrency.
  - [ ] Verify encryption and data deletion.

## Discovered During Work

- (None yet)
