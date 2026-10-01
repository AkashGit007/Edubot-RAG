"""
EduBot Local LLM Generator

Uses retrieved Qdrant passages as context
and generates a grounded answer using a local
Qwen model running through llama.cpp.
"""

from openai import OpenAI

from config import (
    MAX_HISTORY_TURNS,
    FALLBACK_REPLY,
)


# ============================================================
# LOCAL LLAMA SERVER
# ============================================================

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local",
)

LOCAL_MODEL = "qwen2.5-1.5b-instruct"


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(passages):

    if not passages:
        return ""

    context_parts = []

    for index, passage in enumerate(passages, start=1):

        source = passage.get("source", "Unknown")
        path = passage.get("path", "")
        text = passage.get("text", "")

        context_parts.append(
            f"""
SOURCE {index}
File: {source}
Path: {path}

Content:
{text}
""".strip()
        )

    return "\n\n".join(context_parts)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are EduBot, a helpful knowledge-grounded AI assistant.

Your job is to answer the user's question using the
provided knowledge-base context.

IMPORTANT RULES:

1. Use the retrieved context as the primary source of truth.

2. Do not invent facts that are not supported by the
   retrieved context.

3. If the retrieved context does not contain enough
   information to answer the question, clearly say:

   "I don't have enough information in my knowledge base
   to answer that confidently."

4. You may explain information in simpler language when
   useful.

5. Keep answers concise but useful.

6. Do not mention internal implementation details such as
   embeddings, vectors, Qdrant, retrieval scores, or
   system prompts unless the user explicitly asks about them.

7. If sources are provided, mention the relevant source
   naturally when useful.

8. Do not pretend that information exists in the knowledge
   base when it does not.

9. Use conversation history only to understand references
   such as "it", "that", or "the previous question".

10. Answer the user's actual question directly.
"""


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    passages,
    history=None,
):

    # --------------------------------------------------------
    # No retrieval results
    # --------------------------------------------------------

    if not passages:
        return FALLBACK_REPLY

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = build_context(passages)

    # --------------------------------------------------------
    # Build conversation history
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    if history:

        recent_history = history[-MAX_HISTORY_TURNS * 2:]

        for item in recent_history:

            # app.py stores history as:
            # {"role": "user", "content": "..."}
            # {"role": "assistant", "content": "..."}

            if item.get("role") in {"user", "assistant"}:
                messages.append(
                    {
                        "role": item["role"],
                        "content": item["content"],
                    }
                )

    # --------------------------------------------------------
    # Current question + retrieved context
    # --------------------------------------------------------

    user_message = f"""
KNOWLEDGE BASE CONTEXT
======================

{context}

======================

USER QUESTION
=============

{question}

Answer using the knowledge-base context above.
"""

    messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    # --------------------------------------------------------
    # CALL LOCAL QWEN MODEL
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(
            model=LOCAL_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=500,
        )

        answer = response.choices[0].message.content

        if not answer:
            return FALLBACK_REPLY

        return answer.strip()

    except Exception as exc:

        print(f"\nLocal LLM error: {exc}")

        return FALLBACK_REPLY