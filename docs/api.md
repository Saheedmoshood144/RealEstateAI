# RealEstateAI API Documentation

RealEstateAI exposes its machine-learning and conversational capabilities through a REST API built with FastAPI.

The API provides:

- Property-value prediction
- Property recommendation/search
- Natural-language conversational interaction
- Multi-turn conversational sessions
- Structured JSON responses suitable for web or application integration

---

## 1. Running the API

From the project root:

```powershell
python -m uvicorn src.api.app:app --reload

The API will be available at:

http://127.0.0.1:8000
Interactive API Documentation

FastAPI automatically provides interactive Swagger documentation at:

http://127.0.0.1:8000/docs

A ReDoc interface is also available at:

http://127.0.0.1:8000/redoc
2. API Endpoints
Method	Endpoint	Purpose
GET	/	API health/status information
POST	/predict	Predict California property value
POST	/recommend	Search and rank property listings
POST	/chat	Natural-language conversational assistant
3. GET /

Returns basic information confirming that the API is running.

Request
GET /
Example
curl http://127.0.0.1:8000/
Response
{
  "message": "RealEstateAI API is running."
}
4. POST /predict

The /predict endpoint uses the trained California housing regression model to estimate the median house value from eight numerical features.

Required Features

The model expects:

Feature	Description
MedInc	Median income
HouseAge	Median house age
AveRooms	Average number of rooms
AveBedrms	Average number of bedrooms
Population	Population
AveOccup	Average household occupancy
Latitude	Geographic latitude
Longitude	Geographic longitude
Request
POST /predict
Content-Type: application/json
Example
{
  "MedInc": 5.0,
  "HouseAge": 20.0,
  "AveRooms": 6.0,
  "AveBedrms": 1.0,
  "Population": 1500.0,
  "AveOccup": 3.0,
  "Latitude": 34.0,
  "Longitude": -118.0
}
PowerShell Example
$body = @{
    MedInc = 5.0
    HouseAge = 20.0
    AveRooms = 6.0
    AveBedrms = 1.0
    Population = 1500.0
    AveOccup = 3.0
    Latitude = 34.0
    Longitude = -118.0
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
Example Response
{
  "predicted_house_value": 3.XXXX
}

The prediction is represented in the model's original target scale.

The conversational layer converts this value into a dollar estimate when presenting it to the user.

5. POST /recommend

The /recommend endpoint searches the real-estate listings dataset and returns properties matching the supplied filters.

Supported Filters

The recommendation API supports filters including:

City
Neighborhood
Property type
Condition
Minimum bedrooms
Minimum bathrooms
Maximum price
Minimum square footage
Maximum square footage
Number of recommendations
Example Request
POST /recommend
Content-Type: application/json
{
  "city": "Riverside",
  "property_type": "Townhouse",
  "min_bedrooms": 3,
  "max_price": 500000,
  "top_n": 5
}
PowerShell Example
$body = @{
    city = "Riverside"
    property_type = "Townhouse"
    min_bedrooms = 3
    max_price = 500000
    top_n = 5
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/recommend" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
Example Response
{
  "results": [
    {
      "ListingID": "L00176",
      "City": "Riverside",
      "Neighborhood": "Millbrook",
      "PropertyType": "Townhouse",
      "Condition": "Good",
      "Bedrooms": 3,
      "Bathrooms": 1.5,
      "SqFt": 1519,
      "ListPrice": 334709,
      "RecommendationScore": 51.65
    }
  ]
}

The actual number of returned records depends on the supplied filters and available listings.

6. POST /chat

The /chat endpoint provides the main conversational interface to RealEstateAI.

Instead of requiring structured filters, users can communicate with the system using natural language.

The conversational layer can handle requests such as:

Find me a 3-bedroom townhouse in Riverside under $500,000
Show me cheaper ones
What about San Diego?
Only show condos
Show me the second property

The system combines:

Deterministic natural-language parsing
Hybrid parser logic
LLM fallback support
Conversation state
Recommendation engine
ML prediction model
Response formatting
7. Basic Chat Request
Request
POST /chat
Content-Type: application/json
{
  "message": "Find me a 3-bedroom townhouse in Riverside under 500000"
}
Example Response
{
  "intent": "recommend_properties",
  "message": "I found 5 properties matching your request...",
  "data": [
    {
      "ListingID": "L00176",
      "City": "Riverside",
      "Neighborhood": "Millbrook",
      "PropertyType": "Townhouse",
      "Condition": "Good",
      "Bedrooms": 3,
      "Bathrooms": 1.5,
      "SqFt": 1519,
      "ListPrice": 334709,
      "RecommendationScore": 51.65
    }
  ]
}

The data field contains the structured results while message contains the human-readable response.

8. Multi-Turn Conversations

RealEstateAI supports conversational follow-ups using a session_id.

A session allows the assistant to maintain context between requests.

For example:

First request
{
  "message": "Find me a 3-bedroom townhouse in Riverside under 500000",
  "session_id": "demo-session"
}

The assistant establishes the current search context.

Follow-up
{
  "message": "Show me the second property",
  "session_id": "demo-session"
}

The assistant uses the previous conversation context to identify the second property.

Another follow-up
{
  "message": "Show me cheaper ones",
  "session_id": "demo-session"
}

The assistant can refine the previous search rather than requiring the user to repeat all previous filters.

9. Example Multi-Turn Conversation
User
Find me a 3-bedroom townhouse in Riverside under $500,000
Assistant
I found 5 properties matching your request.
User
Show me cheaper ones

The assistant refines the previous search.

User
What about San Diego?

The assistant changes the location while retaining relevant previous preferences.

User
Only show condos

The property type is changed to condos.

User
Show me the second property

The assistant returns the second property from the current recommendation context.

10. Session IDs

A session_id identifies a conversational session.

Example:

demo-session

For production applications, clients should generate a unique session identifier for each conversation.

Example:

import uuid

session_id = str(uuid.uuid4())

The same session_id should be sent with subsequent messages belonging to the same conversation.

Starting a new conversation should use a new session ID.

11. Chat Intent Types

The conversational layer supports several intent categories.

recommend_properties

Used when the user wants to search for or discover properties.

Example:

Find me houses in Riverside with at least three bedrooms.
predict_price

Used when the user requests a property-value estimate.

Example:

Estimate the value of this California property.
general_help

Used when the user asks what the assistant can do.

Example:

What can you help me with?
unknown

Used when the system cannot confidently determine the user's request.

The hybrid architecture can use an LLM fallback for requests that the deterministic parser cannot confidently understand.

12. Recommendation Response

Recommendation responses contain both natural-language and structured information.

Example:

{
  "intent": "recommend_properties",
  "message": "I found 5 properties matching your request.",
  "data": [
    {
      "ListingID": "L00176",
      "City": "Riverside",
      "Neighborhood": "Millbrook",
      "PropertyType": "Townhouse",
      "Condition": "Good",
      "Bedrooms": 3,
      "Bathrooms": 1.5,
      "SqFt": 1519,
      "ListPrice": 334709,
      "RecommendationScore": 51.65
    }
  ]
}

This design allows different clients to use the API in different ways.

For example:

A web frontend can render the structured data.
A chatbot can display the message.
A mobile application can use both.
An analytics application can process the structured listing fields.
13. Prediction Response

A successful prediction returns the model output in structured form.

Example:

{
  "predicted_house_value": 3.125
}

The conversational layer presents the prediction as a human-readable estimated dollar value.

14. Validation and Errors

The API validates incoming requests before passing them to the underlying services.

Examples of invalid requests include:

Missing required prediction features
Invalid feature types
Invalid recommendation limits
Invalid square-footage ranges
Empty requests
Missing chat messages

The API returns appropriate HTTP errors rather than silently processing invalid input.

15. Architecture

The API sits between client applications and the RealEstateAI machine-learning services.

Client
  │
  ▼
FastAPI
  │
  ├── /predict
  │      │
  │      ▼
  │   MLflow Model
  │
  ├── /recommend
  │      │
  │      ▼
  │   Recommendation Engine
  │
  └── /chat
         │
         ▼
   Conversation Service
         │
         ├── Intent Parser
         ├── Hybrid Parser
         ├── LLM Fallback
         ├── Conversation State
         ├── Recommendation Engine
         └── ML Prediction Model

The detailed architecture is documented separately in:

architecture.md
16. Gradio Interface

The REST API can also be accessed through the project's Gradio interface.

Run:

python -m src.ui.gradio_app

The local Gradio interface runs at:

http://127.0.0.1:7860

The Gradio application provides a user-friendly conversational interface on top of the same conversation service.

Users can:

Search for properties
Refine recommendations
Ask follow-up questions
Select individual listings
Request property-value estimates
Start new conversations
17. Example API Workflow

A typical client workflow is:

1. Start FastAPI
       ↓
2. Create a session ID
       ↓
3. Send a natural-language request
       ↓
4. Conversation Service parses the request
       ↓
5. Recommendation Engine or ML model handles it
       ↓
6. API returns structured JSON
       ↓
7. Client displays the response
       ↓
8. Send follow-up messages using the same session ID
18. Testing

The API and conversational system are covered by automated tests.

Run the complete test suite:

python -m pytest -v

The test suite covers:

Prediction API
Recommendation API
Chat API
Conversation service
Intent parsing
Hybrid parser
LLM fallback behavior
Follow-up conversations
Response formatting
Recommendation engine

The project currently maintains a comprehensive automated regression suite.

19. Local Development

Clone the repository and create the project environment:

git clone <https://github.com/Saheedmoshood144/RealEstateAI>
cd RealEstateAI

Create and activate the virtual environment:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Run tests:

python -m pytest -v

Start the API:

python -m uvicorn src.api.app:app --reload

Start the Gradio application:

python -m src.ui.gradio_app
20. API Design Principles

RealEstateAI follows several design principles:

Separation of concerns

The API layer does not contain the recommendation or ML logic directly. It delegates those responsibilities to dedicated services.

Reusable services

The same ConversationService powers both API and Gradio interactions.

Structured responses

Responses contain machine-readable data alongside human-readable messages.

Backward compatibility

The conversational layer extends the existing recommendation and prediction systems without replacing their underlying APIs.

Testability

Core components are independently tested and covered by the project's automated test suite.

Graceful fallback

The conversational system uses deterministic parsing first and can fall back to an LLM when the request cannot be confidently interpreted.

21. Future Production Considerations

The current API is designed primarily for local development and portfolio demonstration.

Before production deployment, additional infrastructure could include:

Authentication and authorization
Rate limiting
Persistent conversation storage
Production database
Centralized logging
Monitoring
HTTPS
Containerization
Cloud deployment
Secrets management
Production-grade session storage
API versioning

These concerns are intentionally separated from the core machine-learning and conversational architecture.

Summary

RealEstateAI provides a complete machine-learning application interface rather than exposing a model in isolation.

The API combines:

Machine Learning + Recommendation Systems + NLP + Conversational State + FastAPI + Gradio

This enables users to interact with the real-estate system through both structured API requests and natural-language conversations.