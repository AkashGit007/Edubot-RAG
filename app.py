"""Text chat loop for testing the RAG pipeline.  Run: python app.py"""
from generator_groq_backup import generate_answer
from retriever import Retriever


def main():
    retriever = Retriever()
    history: list[dict] = []
    print("EduBot ready. Type 'quit' to exit.")

    while True:
        question = input("\nStudent: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        passages = retriever.search(question)
        reply = generate_answer(question, passages, history)
        print(f"Robot: {reply}")

        if passages:
            srcs = ", ".join(f"{p['source']}#{p['chunk_id']} ({p['score']:.2f})" for p in passages)
            print(f"  [sources: {srcs}]")

        # Store the plain Q&A only (not the retrieved context) to keep history small.
        history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": reply},
        ]


if __name__ == "__main__":
    main()