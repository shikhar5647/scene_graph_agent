# agents/graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import TypedDictState
from agents.nodes import split_report_node, candidate_extractor_node, llm_enricher_node, llm_verifier_node, aggregator_node

# a minimal state typed dict (can be richer)
class ReportState(TypedDictState):
    report_text: str

def build_graph():
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
    return graph

def run_graph(report_text: str):
    graph = build_graph()
    initial_state = {"report_text": report_text}
    result = graph.invoke(initial_state)
    # result is the final state snapshot; the aggregator returns scene_graph
    final_state = result.state  # depending on langgraph version; adjust if different
    sg = final_state.get("scene_graph", {})
    return sg
