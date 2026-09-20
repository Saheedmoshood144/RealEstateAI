RealEstateAI

An end-to-end conversational real-estate intelligence platform
combining machine learning, recommendation systems, natural-language
understanding, conversational state, and API engineering.

RealEstateAI is a portfolio-focused ML engineering project that turns
housing data and trained machine-learning components into an interactive
application. Users can discover properties, refine recommendations
through natural-language follow-ups, inspect individual listings, and
request California property-value estimates through a FastAPI API or
Gradio interface.

The project emphasizes application-level ML engineering rather than
model training alone: data → ML → recommendation logic → NLP →
conversational state → API → UI → testing.

🚀 Highlights

🏠 Natural-language property discovery and recommendation

🔎 Filtering by city, neighborhood, property type, condition,
bedrooms, bathrooms, price, and square footage

💬 Multi-turn conversations using a session_id

🔄 Follow-up requests such as:

"Show me cheaper ones"

"What about San Diego?"

"Only show condos"

"Show me the second property"

🧠 Hybrid NLP architecture with a deterministic parser and optional
LLM fallback

🎯 Property-reference resolution for previously returned listings

💰 California housing-value prediction using a registered MLflow
model

⚡ FastAPI REST API with automatic OpenAPI/Swagger documentation

🖥️ Gradio conversational interface

🧪 Automated regression testing across the main application layers

📊 MLflow model tracking/registry integration

🏗️ System Architecture



Conversational flow

User
 │
 ▼
FastAPI /chat or Gradio
 │
 ▼
Conversation Service
 │
 ▼
Hybrid Parser
 ├── Deterministic NLP ──► high-confidence structured intent
 │
 └── Optional LLM fallback ──► ambiguous/low-confidence request
 │
 ▼
Intent + Parameters
 ├── Recommendation Engine ──► ranked property listings
 │
 └── MLflow Prediction Model ──► estimated property value
 │
 ▼
Response Formatter
 │
 ▼
User

This separation keeps the core business logic deterministic and testable
while allowing the conversational layer to become more capable without
replacing the underlying recommendation or prediction systems.

🧠 Conversational Intelligence

The assistant uses a hybrid approach.

1. Deterministic NLP

Common real-estate requests are handled locally using intent and
parameter extraction. This provides predictable behavior for supported
requests without requiring an external LLM call.

Examples:

Find me a 3-bedroom townhouse in Riverside under $500,000.
Only show condos.
Find properties in Riverside with at least 2 bathrooms.

2. Optional LLM fallback

When the deterministic parser cannot confidently interpret a request,
the system can delegate structured intent extraction to a configured LLM
provider.

The LLM layer is optional. The core recommendation, prediction, API, and
conversational tests do not depend on having LLM API credits available.

3. Multi-turn state

The conversational service stores session-level context so users can
refine previous results without repeating every requirement.

Example:

User: Find me a 3-bedroom townhouse in Riverside under $500,000.

Assistant: I found 5 matching properties.

User: Show me cheaper ones.

Assistant: [returns a lower-budget recommendation set]

User: Only show condos.

Assistant: [updates the property type while preserving relevant context]

4. Property reference resolution

The assistant can resolve references to the previous result set:

User: Show me the second property.

The service identifies the second listing from the active session and
returns its property details.

🏠 Recommendation Engine

The recommendation engine supports the following filters:

Parameter         Description

city            City-level location
neighborhood    Neighborhood filter
property_type   House, townhouse, condo, etc.
condition       Property condition
min_bedrooms    Minimum bedrooms
min_bathrooms   Minimum bathrooms
max_price       Maximum listing price
min_sqft        Minimum square footage
max_sqft        Maximum square footage
top_n           Number of recommendations

Matching properties are ranked using the project's recommendation
scoring logic.

Example:

Find me a 3-bedroom townhouse in Riverside under $500,000.

Typical response fields include:

{
  "ListingID": "L00176",
  "City": "Riverside",
  "Neighborhood": "Millbrook",
  "PropertyType": "Townhouse",
  "Bedrooms": 3,
  "Bathrooms": 1.5,
  "SqFt": 1519,
  "ListPrice": 334709,
  "RecommendationScore": 51.65
}

💰 ML Property Valuation

The project includes a California housing regression model registered
with MLflow.

The prediction pipeline expects eight numerical features:

MedInc

HouseAge

AveRooms

AveBedrms

Population

AveOccup

Latitude

Longitude

The conversational service validates the required inputs before invoking
the model. If information is missing, it tells the user which features
are still required rather than attempting an invalid prediction.

The application loads the registered model using:

models:/CaliforniaHousingRandomForest/1

This separates model training/registration from model serving.

⚡ API

RealEstateAI exposes the core functionality through FastAPI.

Endpoints

Endpoint       Method   Purpose

/            GET      Application/root response
/predict     POST     Property-value prediction
/recommend   POST     Structured property recommendations
/chat        POST     Natural-language conversational interface

Interactive documentation

After starting the API, open:

http://127.0.0.1:8000/docs

FastAPI automatically provides interactive OpenAPI documentation.

/chat example

Request:

{
  "message": "Find me a 3-bedroom townhouse in Riverside under 500000",
  "session_id": "demo-session"
}

A follow-up can reuse the same session:

{
  "message": "Show me the second property",
  "session_id": "demo-session"
}

This allows the conversational layer to resolve references against the
previous recommendation results.

🖥️ Gradio Interface

The project includes a Gradio interface for interacting with the
assistant without manually constructing API requests.

Start it with:

python -m src.ui.gradio_app

Then open:

http://127.0.0.1:7860

The Gradio application supports:

Natural-language property search

Multi-turn conversation

Session-based follow-ups

Property selection

Example prompts

New-conversation reset

Connection to the same ConversationService used by the API

Demo screenshots

These paths assume the corresponding screenshots are stored in
docs/screenshots/.

Gradio Interface



Property Recommendation



Multi-Turn Conversation



Property Details



📸 Example Conversation

User:
Find me a 3-bedroom townhouse in Riverside under $500,000.

Assistant:
I found 5 properties matching your request.

User:
Show me cheaper ones.

Assistant:
[returns a refined recommendation set]

User:
What about San Diego?

Assistant:
[updates the location while preserving the relevant conversation context]

User:
Only show condos.

Assistant:
[updates the property type]

User:
Show me the second property.

Assistant:
[returns the second listing from the active result set]

🧪 Testing

The project uses pytest to protect the main application layers.

Run the complete suite:

python -m pytest -v

The latest validated project baseline recorded during development
contains:

103 passed

The test suite covers:

Recommendation engine

Property filtering

Recommendation ranking

Prediction API

Chat API

Intent parsing

Natural-language synonyms

Hybrid parser

LLM fallback behavior

Conversation service

Follow-up requests

Session isolation

Property reference resolution

Response formatting

The test suite is intentionally broad so conversational features can be
added without silently breaking the existing ML and API functionality.

📊 MLflow

MLflow is used to track and serve the registered California housing
model.

Registered model:

CaliforniaHousingRandomForest

Model URI:

models:/CaliforniaHousingRandomForest/1

The application consumes the registered model rather than coupling the
API directly to a training script.

🛠️ Technology Stack

Machine Learning

Python

pandas

NumPy

scikit-learn

MLflow

NLP / Conversational AI

Deterministic natural-language parsing

Hybrid intent parsing

Optional LLM fallback

Session-aware conversational state

Follow-up request resolution

Property-reference resolution

Backend

FastAPI

Uvicorn

Pydantic

User Interface

Gradio

Testing

pytest

API tests

Recommendation-engine tests

Intent-parser tests

Hybrid-parser tests

Conversation-service tests

Follow-up tests

Response-formatting tests

LLM fallback tests

Development

Git

GitHub

VS Code

Python virtual environments

📁 Project Structure

RealEstateAI/
│
├── data/
│   └── raw/
│       └── housing_market.csv
│
├── docs/
│   ├── architecture.png
│   └── screenshots/
│       ├── 01-gradio-home.png
│       ├── 02-property-recommendation.png
│       ├── 03-multiturn-conversation.png
│       └── 04-property-detail.png
│
├── src/
│   ├── api/
│   │   └── app.py
│   │
│   ├── conversation/
│   │   ├── follow_up.py
│   │   ├── formatter.py
│   │   ├── hybrid_parser.py
│   │   ├── intent_parser.py
│   │   ├── llm_fallback.py
│   │   ├── provider.py
│   │   ├── schemas.py
│   │   └── service.py
│   │
│   ├── models/
│   │   └── recommendation_engine.py
│   │
│   └── ui/
│       └── gradio_app.py
│
├── tests/
│   ├── test_api.py
│   ├── test_chat_api.py
│   ├── test_conversation_service.py
│   ├── test_follow_up.py
│   ├── test_hybrid_parser.py
│   ├── test_intent_parser.py
│   ├── test_llm_fallback.py
│   ├── test_recommendation_api.py
│   └── test_response_formatter.py
│
├── pyproject.toml
├── requirements.txt
└── README.md

⚙️ Installation

1. Clone the repository

git clone https://github.com/Saheedmoshood144/RealEstateAI.git
cd RealEstateAI

2. Create a virtual environment

python -m venv .venv

3. Activate the environment

Windows PowerShell:

.venv\Scripts\Activate.ps1

4. Install dependencies

If the repository uses requirements.txt:

python -m pip install -r requirements.txt

Or, if the package is configured for editable installation:

python -m pip install -e .

▶️ Run the FastAPI Application

python -m uvicorn src.api.app:app --reload

Open:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs

▶️ Run the Gradio Application

In a separate terminal:

python -m src.ui.gradio_app

Open:

http://127.0.0.1:7860

The Gradio application and FastAPI application both use the same
underlying conversational service architecture.

🔍 Example Use Cases

Property discovery

Find me a townhouse in Riverside with 3 bedrooms and 2 bathrooms.

Budget filtering

Find properties in Riverside under $500,000.

Property type

Only show condos.

Location follow-up

What about San Diego?

Relative filtering

Show me cheaper ones.

Listing selection

Show me the second property.

Property valuation

I want to estimate the value of a California property.

🔐 Configuration and LLM Fallback

The deterministic conversational pipeline works without an external LLM
provider.

The optional LLM fallback requires a configured provider and valid API
access.

For local development, keep credentials in environment variables rather
than committing secrets to Git:

$env:OPENAI_API_KEY="your_api_key_here"

Never commit API keys, .env files containing secrets, credentials, or
private tokens to GitHub.

If no valid LLM access is available, the system can still operate using
the deterministic conversational components and tested fallback
behavior.

⚠️ Current Limitations

RealEstateAI is a portfolio/research application, not a production
real-estate marketplace.

Current limitations include:

The listing dataset is a project dataset rather than a live
commercial property feed.

Listing prices and availability should not be interpreted as
real-time market data.

The recommendation engine is a portfolio implementation and is not
professional real-estate advice.

Property valuation depends on the features expected by the trained
California housing model.

The LLM fallback depends on the configured provider and available
API access.

The Gradio interface is currently intended for local/demo use unless
separately deployed.

🔮 Future Improvements

Potential future development includes:

Live property-listing integrations

Geospatial search and distance-based recommendations

Embedding-based semantic property search

More advanced conversational memory

Personalized recommendation profiles

Property similarity models

Explainable recommendation scores

Model monitoring

Automated model retraining

Cloud deployment

Authentication and rate limiting

Production frontend/mobile interface

Richer property-level valuation data

🎯 Portfolio Objective

RealEstateAI was built as an end-to-end demonstration of practical ML
engineering.

The project intentionally goes beyond a standalone notebook or trained
model:

Data
  ↓
Machine Learning
  ↓
Recommendation System
  ↓
Natural-Language Processing
  ↓
Conversational State
  ↓
Optional LLM Integration
  ↓
API Engineering
  ↓
Gradio Interface
  ↓
MLflow
  ↓
Automated Testing
  ↓
End-to-End Application

This makes the project a demonstration of how machine-learning
functionality can be packaged into a usable software system.

👨‍💻 Author

Saheed Adewunmi

Machine Learning Engineer | AI Engineer

GitHub: https://github.com/Saheedmoshood144

LinkedIn: https://linkedin.com/in/adewunmi-saheed-729246236

📄 License

This project is intended for educational, portfolio, and demonstration
purposes.

If a formal open-source license is added later, replace this section
with the corresponding license and terms.