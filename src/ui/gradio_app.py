from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

import gradio as gr

from src.conversation.service import ConversationService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LISTINGS_PATH = PROJECT_ROOT / "data" / "raw" / "housing_market.csv"

if not LISTINGS_PATH.exists():
    raise FileNotFoundError(f"Listings file not found: {LISTINGS_PATH}")

conversation_service = ConversationService(listings_path=LISTINGS_PATH)

def create_session_id() -> str:
    """Create a unique session identifier for a Gradio conversation."""
    return str(uuid.uuid4())

def format_response(result: dict[str, Any]) -> str:
    """Convert structured response into text for chatbot."""
    return str(result.get("message", ""))

def chat(message: str, history: list, session_id: str):
    """Send user message through ConversationService."""
    if not message or not message.strip():
        return history, ""

    clean_message = message.strip()
    result = conversation_service.process_message(
        clean_message,
        session_id=session_id,
    )
    assistant_message = format_response(result)

    updated_history = history + [
        {"role": "user", "content": clean_message},
        {"role": "assistant", "content": assistant_message},
    ]
    return updated_history, ""

def new_conversation():
    """Clear chat and create a new backend session."""
    return [], create_session_id()

EXAMPLES = [
    ["Find me a 3-bedroom townhouse in Riverside under $500,000"],
    ["Show me cheaper ones"],
    ["What about San Diego?"],
    ["Only show condos"],
    ["Show me the second property"],
    ["Estimate the value of a California property"],
]

with gr.Blocks(title="CALIFORNIA RealEstateAI") as demo:
    session_state = gr.State(create_session_id())

    gr.Markdown(
        """
        # 🏠 RealEstateAI
        ### Intelligent California Real Estate Assistant
        Ask about properties, refine recommendations conversationally,
        or request a California property-value estimate.
        **Try natural language — no rigid search form required.**
        """
    )

    chatbot = gr.Chatbot(label="RealEstateAI Assistant", height=500)

    with gr.Row():
        message_box = gr.Textbox(
            placeholder="Example: Find me a 3-bedroom townhouse in Riverside under $500,000",
            label="Your message",
            scale=8,
        )
        send_button = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        new_chat_button = gr.Button("🔄 New Conversation")
        clear_button = gr.Button("🗑️ Clear Chat")

    gr.Markdown("### Try an example")
    gr.Examples(examples=EXAMPLES, inputs=message_box)

    gr.Markdown(
        """
        ---
        ### Capabilities
        🏘️ Property recommendations | 💰 Price filtering | 🛏️ Bedroom filtering
        📍 Location search | 🏠 Property-type filtering | 💬 Multi-turn | 📊 ML valuation
        """
    )

    send_button.click(
        fn=chat,
        inputs=[message_box, chatbot, session_state],
        outputs=[chatbot, message_box],
    )
    message_box.submit(
        fn=chat,
        inputs=[message_box, chatbot, session_state],
        outputs=[chatbot, message_box],
    )
    new_chat_button.click(
        fn=new_conversation, inputs=[], outputs=[chatbot, session_state]
    )
    clear_button.click(fn=lambda: [], inputs=[], outputs=[chatbot])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)