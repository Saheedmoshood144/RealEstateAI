# RealEstateAI Architecture

## Overview

RealEstateAI is an end-to-end machine learning and conversational real-estate intelligence platform focused on California property data.

The system combines:

- Machine learning-based property valuation
- Rule-based property recommendation
- Natural-language intent detection
- Hybrid NLP + LLM fallback
- Multi-turn conversational context
- FastAPI REST APIs
- Gradio user interface
- MLflow model management
- Automated testing

The architecture is designed so that the conversational layer can interact with existing ML components without tightly coupling the user interface to the underlying models.

---

## High-Level Architecture

```text
                         ┌──────────────────────┐
                         │        User          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Gradio UI       │
                         │ Conversational Layer │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │      REST API        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Conversation Service      │
                    └───────────────┬───────────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
          ┌──────────────┐  ┌───────────────┐  ┌───────────────┐
          │ Hybrid NLP   │  │ Follow-up /   │  │ Response      │
          │ + LLM        │  │ Session State │  │ Formatter     │
          └──────┬───────┘  └───────────────┘  └───────────────┘
                 │
                 ▼
       ┌─────────────────────┐
       │ Intent & Parameters  │
       │ Extraction           │
       └──────────┬──────────┘
                  │
          ┌───────┴──────────────┐
          │                      │
          ▼                      ▼
 ┌───────────────────┐   ┌─────────────────────┐
 │ Recommendation    │   │ Property Valuation  │
 │ Engine            │   │ ML Model            │
 └─────────┬─────────┘   └──────────┬──────────┘
           │                        │
           ▼                        ▼
 ┌───────────────────┐    ┌─────────────────────┐
 │ Housing Listings  │    │ MLflow Model        │
 │ Dataset           │    │ Registry            │
 └───────────────────┘    └─────────────────────┘