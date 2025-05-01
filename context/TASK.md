# AI Voice Assistant - Task List (Based on REQUIREMENTS.md & architecture.md)

# Last Updated: 2024-07-28

## Phase 1: Core Backend & API Setup

- [x] **Project Setup:** Create basic FastAPI app instance in `AI-Voice-Assistant-/api.py`. (Completed: 2024-07-28)
- [x] **Project Setup:** Create `requirements.txt` in project root. (Completed: 2024-07-28)
- [x] **Project Setup:** Add initial dependencies to `requirements.txt`. (Completed: 2024-07-28)
- [x] **Project Setup:** Install initial dependencies (`pip install -r requirements.txt`). (Completed: 2024-07-28)
- [x] **Project Setup:** Set up `.env` file structure and add initial keys (User confirmed). (Completed: 2024-07-28)
- [x] **Project Setup:** Create Python virtual environment (`venv`). (Completed: 2024-07-28)
- [x] **API Endpoint Stubs:** Create stub functions/routes in `api.py` for required endpoints. (Completed: 2024-07-28)
- [x] **Project Setup:** Verify `.gitignore` includes `venv/`, `__pycache__/`, `.env*`. (Completed: 2024-07-28)
- [ ] **Authentication (Firebase):**
  - [x] Initialize Firebase Admin SDK in `api.py`. (Completed: 2024-07-28)
  - [ ] Implement basic user signup endpoint (e.g., `/auth/signup`).
  - [ ] Implement basic user login endpoint (e.g., `/auth/login`).
  - [x] Add FastAPI dependency/middleware for verifying Firebase ID tokens. (Completed: 2024-07-28)
- [ ] **Database Setup (PostgreSQL):**
  - [x] Set up DB connection logic (using environment variables). (Completed: 2024-07-28)
  - [x] Define initial DB models (e.g., User, CallSession) using SQLModel. (Completed: 2024-07-28)
  - [ ] Implement basic DB interaction for user creation/retrieval.

## Phase 2: Real-time Voice Handling

- [ ] **WebSocket Endpoint:**
  - [ ] Implement WebSocket endpoint (`/ws`) in `api.py`. _(Stub exists)_
  - [ ] Handle WebSocket connection/disconnection. _(Basic handling exists)_
  - [ ] Add authentication check for WebSocket connections (using Firebase token).
- [ ] **Deepgram STT Integration:**
  - [ ] Establish connection to Deepgram Streaming STT endpoint from the backend WebSocket handler.
  - [ ] Forward audio chunks received from the client WebSocket to Deepgram STT.
  - [ ] Receive transcription results from Deepgram STT.

## Phase 3: RAG & LLM Integration

- [ ] **Pinecone Setup:**
  - [ ] Implement Pinecone client initialization.
  - [ ] Create function to query Pinecone index based on transcript text.
- [ ] **OpenAI LLM Call:**
  - [ ] Implement OpenAI client initialization.
  - [ ] Create function to format prompt (transcript + Pinecone results + user context/session memory) and call GPT-4.
- [ ] **RAG Query Logic (`/api/rag/query` or WebSocket flow):**
  - [ ] Integrate transcript reception -> Pinecone query -> LLM call -> response generation.
  - [ ] Implement basic session memory (e.g., store conversation history in DB or cache linked to session ID).

## Phase 4: TTS & Response Delivery

- [ ] **Deepgram TTS Integration:**
  - [ ] Implement function to call Deepgram TTS with LLM response text.
  - [ ] Stream TTS audio received from Deepgram back to the client via the WebSocket.
- [ ] **Connect RAG to TTS:**
  - [ ] Trigger TTS generation after receiving the final LLM response.

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
  - [ ] Set up `/tests` directory.
  - [ ] Write unit tests for authentication, API endpoints, RAG logic, etc.
- [ ] **Non-Functional Requirements:**
  - [ ] Test latency (< 1.5s).
  - [ ] Test concurrency.
  - [ ] Verify encryption and data deletion.

## Discovered During Work

- (None yet)
