This document outlines all functional and non-functional requirements for the AI Voice Agent MVP for use with Cursor AI or other autonomous coding agents.

## Project Summary

A real-time web-based AI voice agent for SaaS users to ask account-related questions. Uses DeepGram for voice input/output and a Pinecone-backed RAG system to generate answers using company documentation and user data.

## Functional Requirements

- [ ] Users must log in to the system before making calls.
- [ ] Users must be able to initiate a voice call via browser.
- [ ] AI must answer account-related questions using:
  - [ ] Preloaded company KB (FAQs, docs)
  - [ ] Account-specific context (recent activity, billing info)
- [ ] Real-time STT and TTS (DeepGram)
- [ ] Voice agent must maintain session memory during calls
- [ ] Calls and queries must be logged (30-day rolling retention)
- [ ] Admin dashboard must display:
  - [ ] Number of calls per day
  - [ ] Average call duration
  - [ ] Top repeated questions
  - [ ] Fallback rate (questions AI couldn’t answer)
- [ ] Admin should be able to export analytics as CSV

## Non-Functional Requirements

- [ ] Voice latency must be < 1.5 seconds round-trip
- [ ] AI responses must be streamed within 1.5 seconds of end-of-speech
- [ ] System must scale to 50 concurrent users (MVP limit)
- [ ] Privacy: session logs are auto-deleted after 30 days
- [ ] Data must be encrypted in transit and at rest
