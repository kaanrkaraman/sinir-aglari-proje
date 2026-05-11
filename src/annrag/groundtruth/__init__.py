"""Ground-truth dataset construction (Milestone 2).

The dataset drives every later experiment: retrieval-only metrics
(MRR/NDCG/Recall@k) and RAGAS faithfulness/relevance scoring all read from
the same JSONL. Records pin questions to *page-level* `relevant_doc_ids`
(Wikivoyage article titles) so they remain valid across chunking strategies.
"""
