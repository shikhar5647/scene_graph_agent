# agents/graph.py
from typing import TypedDict, Optional, Dict, Any
from agents.nodes import (
    split_report_node,
    candidate_extractor_node,
    llm_enricher_node,
    llm_verifier_node,
    aggregator_node,
)

# a minimal state typed dict (can be richer)
class ReportState(TypedDict):
    report_text: str
    sentences: Optional[list[str]]
    candidates: Optional[dict]
    scene_graph_partial: Optional[dict]
    scene_graph: Optional[dict]


def _run_nodes_sequential(report_text: str) -> Dict[str, Any]:
    """Fallback runner that executes the nodes sequentially without langgraph.

    This is used when the installed langgraph API is incompatible or if
    the StateGraph approach raises errors (for example, unknown-node edges).
    It mirrors the same pipeline order used previously.
    """
    state: Dict[str, Any] = {"report_text": report_text}

    # 1. Split
    out = split_report_node(state)
    state.update(out)

    # 2. Candidate extraction
    out = candidate_extractor_node(state)
    state.update(out)

    # 3. LLM enrichment
    out = llm_enricher_node(state)
    state.update(out)

    # 4. LLM verification/normalization
    out = llm_verifier_node(state)
    state.update(out)

    # 5. Aggregation
    out = aggregator_node(state)
    state.update(out)

    return state.get("scene_graph", {})


def run_graph(report_text: str):
    """Attempt to build and run a langgraph StateGraph if available, otherwise
    fall back to a simple sequential runner.
    """
    try:
        # try to import StateGraph lazily (may fail on incompatible langgraph)
        from langgraph.graph import StateGraph, START, END

        graph = StateGraph(ReportState)
        # add nodes (functions)
        graph.add_node(split_report_node, name="split_report")
        graph.add_node(candidate_extractor_node, name="candidate_extractor")
        graph.add_node(llm_enricher_node, name="llm_enricher")
        graph.add_node(llm_verifier_node, name="llm_verifier")
        graph.add_node(aggregator_node, name="aggregator")
        # edges / flow
        graph.add_edge(START, "split_report")
        graph.add_edge("split_report", "candidate_extractor")
        graph.add_edge("candidate_extractor", "llm_enricher")
        graph.add_edge("llm_enricher", "llm_verifier")
        graph.add_edge("llm_verifier", "aggregator")
        graph.add_edge("aggregator", END)
        graph = graph.compile()
        initial_state = {"report_text": report_text}
        result = graph.invoke(initial_state)
        final_state = getattr(result, "state", result)
        return final_state.get("scene_graph", {})
    except Exception:
        # Any failure in building/running the StateGraph falls back to sequential
        return _run_nodes_sequential(report_text)
