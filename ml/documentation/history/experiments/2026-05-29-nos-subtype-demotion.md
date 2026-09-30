# NOS subtype demotion (keyword rule)

**Date:** 2026-05-29 · **Verdict:** reverted

## Hypothesis
High-frequency generic "NOS" labels (Carcinoma, Adenoma, Adenocarcinoma, Sarcoma NOS; about 53-63% precision) could be improved by replacing a top-1 NOS term with a specific in-group label when the report text carries subtype keywords, in the style of the Lipoma rescue rule.

## Setup
- Baseline: promoted four-stage `ml/`, test split, G+S 60.1% (Good 45.7%).
- Rule: `_DEMOTION_RULES` and `resolve_subtype_label()` in the subtype-keyword module, hooked into per-case categorisation after stage 3.

## Result
| Metric | Baseline | With demotion | Delta |
|---|---|---|---|
| Good | 45.7% | 43.4% | -2.3 pp |
| Slight | not recorded | +2.5 pp higher | +2.5 pp |
| G+S | 60.1% | 60.3% | +0.2 pp |

Per-case accounting on the test split: 391 demotions, 238 HARM (correct NOS became a wrong specific term), 25 WIN, 119 neutral, 9 both correct. Net Good -213, about ten times more harm than help. Organ-lineage keywords also hurt (for example Hepatocellular carcinoma precision 77% to 67%).

## Why
The annotation ground truth is NOS-biased: a subtype keyword in the report is essentially uncorrelated with NOS being the wrong label. The annotator and cascade keep the generic term even when organ or architecture words appear. Predicting NOS on these cases is correct by the registry's own conventions; the low per-label precision comes from how false positives are counted, not a keyword-fixable model error. Combined G+S stayed flat while Good fell and Slight rose, a bad trade.

## Don't retry unless…
The label side changes: de-bias the NOS preference in the annotation (re-annotation) and retrain. Another keyword set will not help.

## Where it went
Fully reverted, no residue. Recorded so the negative result survives. Related open item: [Adenomas subtype keywords](../open-ideas.md) is a different rule (organ filters inside one group, with the no-match fallback preserved), not a demotion of NOS.
