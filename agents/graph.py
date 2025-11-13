# agents/graph.py
from typing import TypedDict, Optional, Dict, Any
import numpy as np
from agents.nodes import (
    split_report_node,
    candidate_extractor_node,
    llm_enricher_node,
    llm_verifier_node,
    matrix_builder_node,
    aggregator_node,
)

# State definition for the pipeline
class ReportState(TypedDict):
    report_text: str
    sentences: Optional[list[str]]
    candidates: Optional[dict]
    findings_dict: Optional[dict]
    verified_findings: Optional[dict]
    scene_graph_matrix: Optional[np.ndarray]
    final_matrix: Optional[np.ndarray]
    metadata: Optional[dict]


def _run_nodes_sequential(report_text: str) -> Dict[str, Any]:
    """Fallback runner that executes the nodes sequentially without langgraph."""
    state: Dict[str, Any] = {"report_text": report_text}

    # 1. Split sentences
    out = split_report_node(state)
    state.update(out)

    # 2. Candidate extraction
    out = candidate_extractor_node(state)
    state.update(out)

    # 3. LLM enrichment
    out = llm_enricher_node(state)
    state.update(out)

    # 4. LLM verification
    out = llm_verifier_node(state)
    state.update(out)

    # 5. Matrix building
    out = matrix_builder_node(state)
    state.update(out)

    # 6. Aggregation
    out = aggregator_node(state)
    state.update(out)

    return {
        "matrix": state.get("final_matrix"),
        "metadata": state.get("metadata")
    }


def run_graph(report_text: str) -> Dict[str, Any]:
    """Run the scene graph extraction pipeline and return matrix + metadata."""
    try:
        from langgraph.graph import StateGraph, START, END

        graph = StateGraph(ReportState)
        
        # Add nodes
        graph.add_node(split_report_node, name="split_report")
        graph.add_node(candidate_extractor_node, name="candidate_extractor")
        graph.add_node(llm_enricher_node, name="llm_enricher")
        graph.add_node(llm_verifier_node, name="llm_verifier")
        graph.add_node(matrix_builder_node, name="matrix_builder")
        graph.add_node(aggregator_node, name="aggregator")
        
        # Define edges
        graph.add_edge(START, "split_report")
        graph.add_edge("split_report", "candidate_extractor")
        graph.add_edge("candidate_extractor", "llm_enricher")
        graph.add_edge("llm_enricher", "llm_verifier")
        graph.add_edge("llm_verifier", "matrix_builder")
        graph.add_edge("matrix_builder", "aggregator")
        graph.add_edge("aggregator", END)
        
        graph = graph.compile()
        initial_state = {"report_text": report_text}
        result = graph.invoke(initial_state)
        
        final_state = getattr(result, "state", result)
        return {
            "matrix": final_state.get("final_matrix"),
            "metadata": final_state.get("metadata")
        }
        
    except Exception:
        # Fallback to sequential execution
        return _run_nodes_sequential(report_text)