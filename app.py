"""
Apple Support Bot — an intermediate-level LangGraph agent deployed on Amazon
Bedrock AgentCore Runtime.

Capabilities beyond the basic calculator example:
  - Two custom tools: troubleshooting lookup + order/warranty lookup
  - Multi-turn conversation: short-term memory (recent turns in this session)
    is replayed into the LLM context on every call
  - Long-term memory: facts and device preferences are extracted across
    sessions using AgentCore Memory strategies, so the bot can recall things
    the user told it in a *previous, separate* conversation/day, keyed by
    actor_id (not just session_id)
"""
import os
from typing import Annotated, Any, Dict, List, TypedDict

from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole, RetrievalConfig

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# ---------------------------------------------------------------------------
# 1. LLM
# ---------------------------------------------------------------------------
llm = ChatBedrock(
    model_id="amazon.nova-lite-v1:0",
    region_name=AWS_REGION,
    model_kwargs={"temperature": 0},
)

# ---------------------------------------------------------------------------
# 2. Tools
# ---------------------------------------------------------------------------
_TROUBLESHOOTING_KB = {
    "iphone battery": (
        "1) Check Settings > Battery > Battery Health & Charging. "
        "2) Update to the latest iOS. "
        "3) If capacity is below 80%, book a battery replacement at an Apple Store."
    ),
    "mac wont turn on": (
        "1) Force restart: hold the power button for 10 seconds. "
        "2) Check the charger and cable on a different outlet. "
        "3) Try an SMC reset (Intel Macs) or NVRAM reset. "
        "4) If still nothing, book a Genius Bar appointment."
    ),
    "airpods not connecting": (
        "1) Put both AirPods in the case, close the lid for 15s, then reopen near your iPhone. "
        "2) Forget the device in Bluetooth settings and re-pair. "
        "3) Reset AirPods: hold the setup button on the case for 15s until the light flashes amber then white."
    ),
    "ipad frozen": (
        "1) Force restart: press Volume Up, then Volume Down, then hold the Top button "
        "until the Apple logo appears. "
        "2) Update iPadOS once it boots. "
        "3) If unresponsive, connect to a computer and restore via Finder."
    ),
    "apple watch not charging": (
        "1) Make sure the magnetic charger is fully snapped on and centered. "
        "2) Clean the charging contacts on the watch and cable. "
        "3) Try a different USB power source. "
        "4) If it still won't charge after 30 minutes, contact Apple Support for a hardware check."
    ),
}


@tool
def troubleshoot_lookup(issue_keyword: str) -> str:
    """Look up official troubleshooting steps for a known Apple product issue.
    Pass a short keyword describing the problem, e.g. 'iphone battery',
    'mac wont turn on', 'airpods not connecting', 'ipad frozen',
    'apple watch not charging'."""
    key = issue_keyword.lower().strip()
    for topic, steps in _TROUBLESHOOTING_KB.items():
        if topic in key or key in topic:
            return f"Troubleshooting steps for '{issue_keyword}':\n{steps}"
    available = ", ".join(_TROUBLESHOOTING_KB.keys())
    return f"No exact match found for '{issue_keyword}'. Known topics: {available}"


_MOCK_ORDERS = {
    "W1234567890": {"device": "iPhone 15 Pro", "purchase_date": "2025-03-10", "applecare": False},
    "W1234567891": {"device": "MacBook Air M3", "purchase_date": "2024-11-02", "applecare": True},
    "W1234567892": {"device": "AirPods Pro 2", "purchase_date": "2026-01-15", "applecare": False},
}


@tool
def order_warranty_lookup(order_id: str) -> str:
    """Look up an Apple order's device, purchase date, and warranty/AppleCare+
    status by order ID. Example valid order IDs: W1234567890, W1234567891,
    W1234567892."""
    import datetime

    order = _MOCK_ORDERS.get(order_id.upper().strip())
    if not order:
        return f"No order found with ID '{order_id}'. Please double-check the order number."

    purchase = datetime.date.fromisoformat(order["purchase_date"])
    warranty_months = 12 * (2 if order["applecare"] else 1)
    total_months = purchase.month - 1 + warranty_months
    expiry = datetime.date(
        purchase.year + total_months // 12,
        total_months % 12 + 1,
        min(purchase.day, 28),
    )
    status = "ACTIVE" if expiry >= datetime.date.today() else "EXPIRED"
    return (
        f"Order {order_id.upper()}: {order['device']}, purchased {order['purchase_date']}. "
        f"AppleCare+: {'Yes' if order['applecare'] else 'No'}. "
        f"Warranty status: {status} (expires {expiry.isoformat()})."
    )


@tool
def update_order_applecare(order_id: str, has_applecare: bool = True) -> str:
    """Update whether an order has AppleCare+ coverage. Call this whenever the
    user tells you they've just purchased (or cancelled) AppleCare+ for an
    existing order, so future warranty lookups reflect the change. Set
    has_applecare=True to add coverage, False to remove it."""
    key = order_id.upper().strip()
    order = _MOCK_ORDERS.get(key)
    if not order:
        return f"No order found with ID '{order_id}'. Cannot update AppleCare status."

    order["applecare"] = has_applecare
    action = "added to" if has_applecare else "removed from"
    return (
        f"AppleCare+ has been {action} order {key} ({order['device']}). "
        f"Future warranty lookups for this order will reflect the new coverage."
    )


tools = [troubleshoot_lookup, order_warranty_lookup, update_order_applecare]
llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------------------------------
# 3. LangGraph StateGraph (ReAct loop, same shape as the calculator example)
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def agent_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
compiled_graph = graph.compile()

SYSTEM_PROMPT = (
    "You are Apple Support Assistant, a helpful, concise agent for troubleshooting "
    "Apple devices and looking up order/warranty status. Use the tools available "
    "to you whenever the user asks about a specific device problem or an order."
)

# ---------------------------------------------------------------------------
# 4. Multi-turn memory (AgentCore Memory: short-term + long-term)
# ---------------------------------------------------------------------------
MEMORY_ID = os.environ.get("APPLE_BOT_MEMORY_ID")
memory_manager = MemorySessionManager(memory_id=MEMORY_ID, region_name=AWS_REGION) if MEMORY_ID else None

RETRIEVAL_CONFIG = {
    "/support/facts/{actorId}/": RetrievalConfig(top_k=3, relevance_score=0.2),
    "/support/preferences/{actorId}/": RetrievalConfig(top_k=3, relevance_score=0.2),
}


def _short_term_history_as_messages(session, k: int = 6) -> List[Dict[str, str]]:
    """Turn the last k stored turns for this session into chat messages."""
    history: List[Dict[str, str]] = []
    try:
        turns = session.get_last_k_turns(k=k)
    except Exception:
        return history
    for turn in turns:
        for msg in turn:
            role = (msg.get("role") or "").lower()
            text = (msg.get("content") or {}).get("text", "")
            if not text:
                continue
            if role == "user":
                history.append({"role": "user", "content": text})
            elif role == "assistant":
                history.append({"role": "assistant", "content": text})
    return history


def _long_term_memories_as_context(memories: List[Dict[str, Any]]) -> str:
    facts = [m.get("content", {}).get("text", "") for m in memories]
    facts = [f for f in facts if f]
    if not facts:
        return ""
    return "Remembered from earlier conversations with this user:\n" + "\n".join(f"- {f}" for f in facts)


def _content_to_text(content: Any) -> str:
    """ChatBedrock messages can return `.content` as a plain string OR as a
    list of content blocks (e.g. reasoning_content + text blocks). Normalize
    either shape down to a single display string, since AgentCore Memory's
    process_turn_with_llm() requires the callback to return a plain str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p.strip() for p in parts if p.strip())
    return str(content)


def _run_agent(user_input: str, history: List[Dict[str, str]], memory_context: str) -> str:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if memory_context:
        messages.append({"role": "system", "content": memory_context})
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    result = compiled_graph.invoke({"messages": messages})
    return _content_to_text(result["messages"][-1].content)


# ---------------------------------------------------------------------------
# 5. AgentCore Runtime entrypoint
# ---------------------------------------------------------------------------
app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict):
    user_prompt = payload.get("prompt", "")
    actor_id = payload.get("actor_id", "guest-user")
    session_id = payload.get("session_id", "default-session")

    if not memory_manager:
        # No memory configured (APPLE_BOT_MEMORY_ID unset) -> stateless single-turn fallback
        answer = _run_agent(user_prompt, history=[], memory_context="")
        return {"result": answer, "memory_enabled": False}

    session = memory_manager.create_memory_session(actor_id=actor_id, session_id=session_id)
    history = _short_term_history_as_messages(session)

    def llm_callback(user_input: str, memories: List[Dict[str, Any]]) -> str:
        memory_context = _long_term_memories_as_context(memories)
        return _run_agent(user_input, history=history, memory_context=memory_context)

    memories, answer, _event = session.process_turn_with_llm(
        user_input=user_prompt,
        llm_callback=llm_callback,
        retrieval_config=RETRIEVAL_CONFIG,
    )

    return {
        "result": answer,
        "memory_enabled": True,
        "actor_id": actor_id,
        "session_id": session_id,
        "long_term_memories_used": len(memories),
    }


if __name__ == "__main__":
    app.run()
