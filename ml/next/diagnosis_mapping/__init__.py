"""Diagnosis mapping: the 3-tier cascade (keyword_tiers, llm_tier), the local
LLM client, the ensemble cleanup pass, versioned silver generations, and
coverage stats. See silver.py for the public entry points (``run``,
``import_legacy``, ``load_silver``)."""
