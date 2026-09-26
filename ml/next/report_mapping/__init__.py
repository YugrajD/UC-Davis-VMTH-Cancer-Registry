"""Report-mapping: sections, model (backbone + heads + generation), inference.

Per-section contrastive PetBERT backbone -> concat-3 (2304-d) case embedding ->
case-presence gate -> group (+ tail gate) -> per-group label-presence ->
keyword correction (+ Lipoma rescue). See ml/documentation/ml-rewrite-plan.md
and ml/documentation/{production-pipeline,classifiers}.md for the design this
package carries forward.
"""
