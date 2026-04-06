from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run
from Task_14 import basic_rag_pipeline

RAG_DOCUMENTS = [
    "LangChain v0.2 introduced LangChain Expression Language (LCEL) for composing chains.",
    "pgvector is a PostgreSQL extension supporting L2, inner product, and cosine distance.",
    "LangSmith provides tracing for every LLM call including token counts and latency.",
    "RAG stands for Retrieval-Augmented Generation and improves factual accuracy of LLMs.",
    "OpenAI's text-embedding-3-small produces 1536-dimensional embedding vectors.",
    "LangChain agents use a ReAct loop: Thought → Action → Observation → Answer.",
]

def run_langsmith_evaluation() -> dict:
    """Evaluates the RAG pipeline on the LangSmith dataset."""

    dataset_name = "rag-eval-dataset"

    # Target function
    def target(inputs: dict) -> dict:
        answer = basic_rag_pipeline(RAG_DOCUMENTS, inputs["question"])
        return {"answer": answer}

    # Custom evaluator
    def contains_correct_answer(run: Run, example: Example) -> dict:
        prediction = run.outputs.get("answer", "").lower()
        reference = example.outputs.get("answer", "").lower()
        score = 1 if reference in prediction else 0
        return {
            "key": "answer_contains_expected",
            "score": score,
        }

    # Run evaluation
    results = evaluate(
        target=target,
        data=dataset_name,
        evaluators=[contains_correct_answer],
        experiment_prefix="rag-challenge-eval",
    )

    # Aggregate results
    num_examples = len(results)
    pass_count = sum(
        1 for r in results
        if r["evaluation"].get("answer_contains_expected") == 1
    )
    pass_rate = pass_count / num_examples if num_examples > 0 else 0.0

    return {
        "dataset": dataset_name,
        "num_examples": num_examples,
        "pass_rate": pass_rate,
    }


if __name__ == "__main__":
    print("\n[Task 20] Run LangSmith Evaluation")
    eval_summary = run_langsmith_evaluation()
    print(f"  Dataset     : {eval_summary['dataset']}")
    print(f"  # Examples  : {eval_summary['num_examples']}")
    print(f"  Pass rate   : {eval_summary['pass_rate']:.2f}")