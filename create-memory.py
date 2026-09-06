"""
One-time provisioning script: creates the Bedrock AgentCore Memory resource
used by the Apple Support Bot, with two long-term memory strategies:

  1. SEMANTIC   -> extracts durable support facts/issues discussed
                   (namespace: /support/facts/{actorId}/)
  2. USER_PREFERENCE -> extracts the user's device(s) and preferences
                   (namespace: /support/preferences/{actorId}/)

Short-term memory (raw conversation turns, for multi-turn continuity within
a session) is included automatically with every AgentCore Memory resource.

Run this once, then copy the printed Memory ID into your deploy command:

    python create_memory.py
    agentcore deploy --env APPLE_BOT_MEMORY_ID=<printed-memory-id>

Re-running this script is safe — it reuses an existing memory with the same
name instead of creating a duplicate.
"""
import os

from bedrock_agentcore.memory import MemoryClient

REGION = os.environ.get("AWS_REGION", "us-east-1")
MEMORY_NAME = "apple_support_bot_memory"

client = MemoryClient(region_name=REGION)

print(f"Creating (or reusing) memory '{MEMORY_NAME}' in {REGION}...")

memory = client.create_or_get_memory(
    name=MEMORY_NAME,
    description="Long-term + short-term memory for the Apple Support multi-turn bot",
    event_expiry_days=90,
    strategies=[
        {
            "semanticMemoryStrategy": {
                "name": "SupportFacts",
                "namespaces": ["/support/facts/{actorId}/"],
            }
        },
        {
            "userPreferenceMemoryStrategy": {
                "name": "UserDevicePreferences",
                "namespaces": ["/support/preferences/{actorId}/"],
            }
        },
    ],
)

memory_id = memory["memoryId"] if "memoryId" in memory else memory["id"]
print("\n✅ Memory ready.")
print(f"   Memory ID: {memory_id}")
print(f"   Status:    {memory.get('status')}")
print("\nNext step — deploy with this memory id:")
print(f'   agentcore deploy --env APPLE_BOT_MEMORY_ID={memory_id}')
