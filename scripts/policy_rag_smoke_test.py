import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas import GraphState, Intent, IntentRouteResult, RerankerMode, RetrievalMode
from app.services import (
    ChatCompletionsModelProvider,
    KnowledgeRetriever,
    ModelProviderError,
    PolicyRAGService,
)
from app.workflows import build_main_graph, build_policy_rag_graph


DEFAULT_MESSAGE = "What does the EYLF say about play-based learning?"


class PolicyRouter:
    def route(self, user_message: str) -> IntentRouteResult:
        return IntentRouteResult(
            intent=Intent.POLICY_QA,
            confidence=1.0,
            reason="Smoke test forces the policy QA route.",
        )


def build_policy_workflow(*, use_real_model: bool, reranker: RerankerMode):
    model_provider = ChatCompletionsModelProvider() if use_real_model else None
    retriever = KnowledgeRetriever()
    service = PolicyRAGService(
        retriever=retriever,
        model_provider=model_provider,
        retrieval_mode=RetrievalMode.BM25,
        reranker=reranker,
        top_k=3,
    )
    return build_policy_rag_graph(service)


def print_final_state(state: GraphState) -> None:
    print("=== Final GraphState ===")
    print(f"intent={state.intent.value}")
    print(f"workflow_status={state.workflow_status.value}")
    print(f"needs_clarification={state.needs_clarification}")
    if state.clarification_question:
        print(f"clarification_question={state.clarification_question}")

    if state.draft:
        print()
        print("=== Policy Answer Draft ===")
        print(f"title={state.draft.title}")
        print(f"is_draft={state.draft.is_draft}")
        print(state.draft.content)

    if state.citations:
        print()
        print("=== Citations ===")
        for index, citation in enumerate(state.citations, start=1):
            print(
                f"[{index}] source={citation.source} "
                f"title={citation.title} "
                f"section={citation.section} "
                f"page={citation.page} "
                f"url={citation.url}"
            )

    if state.errors:
        print()
        print("=== Errors ===")
        for error in state.errors:
            print(json.dumps(error.model_dump(mode="json"), indent=2, ensure_ascii=False))

    print()
    print("=== Trace ===")
    for event in state.trace:
        print(f"- {event.step}: {event.message}")
        if event.metadata:
            print(json.dumps(event.metadata, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the policy RAG graph smoke test.")
    parser.add_argument(
        "message",
        nargs="?",
        default=DEFAULT_MESSAGE,
        help="Policy question to send through the main graph.",
    )
    parser.add_argument(
        "--real-model",
        action="store_true",
        help="Use ChatCompletionsModelProvider for final answer generation.",
    )
    parser.add_argument(
        "--reranker",
        choices=[mode.value for mode in RerankerMode],
        default=RerankerMode.LEXICAL.value,
        help="Reranker to use for the smoke test. Defaults to lexical for offline stability.",
    )
    args = parser.parse_args()

    print("POLICY_RAG_SMOKE_START")
    print(f"message={args.message!r}")
    print(f"real_model={args.real_model}")
    print(f"reranker={args.reranker}")

    try:
        graph = build_main_graph(
            router=PolicyRouter(),
            policy_workflow=build_policy_workflow(
                use_real_model=args.real_model,
                reranker=RerankerMode(args.reranker),
            ),
        )
        result = graph.invoke(
            GraphState(
                request_id="policy-rag-smoke",
                session_id="policy-rag-smoke-session",
                user_message=args.message,
            )
        )
        final_state = GraphState.model_validate(result)
    except ModelProviderError as error:
        print("POLICY_RAG_SMOKE_FAILED")
        print(json.dumps(error.to_dict(), indent=2, ensure_ascii=False))
        return 1

    print_final_state(final_state)
    print()
    print("POLICY_RAG_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
