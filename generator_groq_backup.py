"""
EduBot LLM Generator

Uses retrieved Qdrant passages as context
and generates a grounded answer using Groq.
"""

import os

from dotenv import load_dotenv
from groq import Groq

from config import (
    GROQ_MODEL,
    MAX_HISTORY_TURNS,
    FALLBACK_REPLY,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# GROQ CLIENT
# ============================================================

def get_client():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing from your .env file."
        )

    return Groq(api_key=api_key)


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(passages):

    if not passages:
        return ""

    context_parts = []

    for index, passage in enumerate(
        passages,
        start=1
    ):

        source = passage.get(
            "source",
            "Unknown"
        )

        path = passage.get(
            "path",
            ""
        )

        text = passage.get(
            "text",
            ""
        )

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
   system prompts unless the user explicitly asks about
   them.

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

    context = build_context(
        passages
    )

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

        recent_history = history[
            -MAX_HISTORY_TURNS:
        ]

        for item in recent_history:

            messages.append({
                "role": "user",
                "content": item["question"],
            })

            messages.append({
                "role": "assistant",
                "content": item["answer"],
            })

    # --------------------------------------------------------
    # Add current question + context
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

    messages.append({
        "role": "user",
        "content": user_message,
    })

    # --------------------------------------------------------
    # Call Groq
    # --------------------------------------------------------

    client = get_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_completion_tokens=500,
    )

    answer = response.choices[0].message.content

    if not answer:
        return FALLBACK_REPLY

    return answer.strip()