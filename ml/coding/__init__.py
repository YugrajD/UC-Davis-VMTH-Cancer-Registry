"""The coding rule, combined codes, corrected annotations and the review queue.

Implements "Coding a case" in ml/documentation/icd-mapping-strategy.md:
gold > silver > bronze precedence, a vagueness table read from the diagnosis
mapping's ``decision_stage``, and the review queue that routes vague/
low-confidence cases to the specialist.
"""
