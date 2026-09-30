# Open ideas

Ideas that were proposed for report mapping but never tried, restated against the current code (`ml/report_mapping/`, `ml/taxonomy/`). None is scheduled. Before acting on one, read the experiment records it links to, re-measure the baseline (the old group-level figures came from an earlier architecture), and show Good and Slight separately.

Two general rules from the closed experiments apply to all of them:
- Anything that changes the embeddings (text, backbone) invalidates every head; train into `candidate/` and promote through the gold-eval rule ([decision 0004](decisions/0004-generations-candidate-current-archive.md)).
- Gold-eval does not exist yet, so a real promotion cannot happen; until then, results can only be silver-eval on the calibration or test partition.

## Rule and threshold ideas (no retraining)

### Adenomas subtype keywords (was QW4)
The largest group ("Adenomas and adenocarcinomas", about 17.6% of the old test set, 18.1% Good then) has no entry in the subtype rules (`ml/taxonomy/subtype.py`; the seven groups with rules are Mast cell, Blood vessel, Melanocytoma and Melanomas, Meningiomas, Lipomatous, Osseous and chondromatous, Gliomas). The idea is a set of organ markers (mammary, apocrine, sebaceous, hepatoid, perianal, thyroid, ceruminous, biliary, pancreatic, renal, prostatic and similar) that narrows the label pool before term selection.
- Keep the existing contract: if no keyword matches, or the match empties the pool, return the pool unchanged.
- Curate markers against `ml/taxonomy/labels.csv`; test by comparing Good and Slight for this group on the calibration partition.
- Caution: [NOS demotion](experiments/2026-05-29-nos-subtype-demotion.md) failed because the annotation prefers NOS even when subtype words appear, and organ words hurt there too. This idea filters inside a group and never demotes NOS, but the same bias could show up.
- Old estimate: +1.5 to +2.5 pp G+S (unverified, from a different architecture).

### Per-group group-head thresholds (was QW5)
The group head uses one threshold for all about 25 groups (calibration fits gate, group and tail jointly; only label-presence thresholds are per group). Soft tissue tumors and sarcomas over-predicted (F1 0.63, precision 0.50) in the old 17-group analysis. The idea is a per-group threshold dict, starting with that group, or a capped `pos_weight` for it at retrain.
- Re-measure first: that figure predates concat-3.
- Warning: per-group thresholds on a global label-presence model were exhausted and redundant with argmax ([GLP record](experiments/2026-05-26-glp-listwise-and-per-group-thresholds.md)); the group head is a different model, so it is not ruled out, but expect small gains.

## Model ideas (retraining)

### Bilinear or low-rank label-presence head (was MI1)
Replace the per-group `[section | label] -> hidden -> 1` MLP with a low-rank bilinear score (`W = U V^T`, rank 64 or less) to get pairwise structure with fewer parameters. Add it as a new head type, not a branch inside the existing trainer, and keep loading old checkpoints. Not started; the current head definitions are in `ml/report_mapping/model/heads.py`. Hard-negative and threshold-only changes to these heads have failed ([QW1/QW2](experiments/2026-05-10-lp-hard-negatives-qw1-qw2.md), [retry](experiments/2026-05-13-lp-hard-neg-retry-and-top3-rerank.md)), so an architecture change is the remaining lever.

### Per-group softmax label-presence head (was MI2)
One softmax over `(labels in the group + "none")` per group instead of independent sigmoids, so scores are calibrated against each other and thresholds matter less. Supporting evidence: on the global model, listwise softmax cross-entropy fixed the score saturation that per-pair BCE caused ([GLP record](experiments/2026-05-26-glp-listwise-and-per-group-thresholds.md)). Risk: the "none" class (case is in the group but the gold label is outside the taxonomy) is easy to mis-specify; build a small audit set first.

### Split the Adenomas group (was MI3)
Split "Adenomas and adenocarcinomas" by anatomic site (mammary, cutaneous adnexal, gastrointestinal, endocrine) at the group head, only if the adenoma keywords above are not enough. Enforce at least 300 train cases per child. The earlier failure was an upward expansion (promoting rare groups; [record](experiments/2026-05-07-uncommon-threshold-and-27-groups.md)); this is a downward split of a large group, which is untested. Needs a taxonomy remap, a group-head retrain and retrains of the affected per-group heads, then a new generation.

### Generative label decoder (was LB1)
Replace stage 3 with a small sequence-to-sequence decoder (for example Flan-T5-small) constrained to the label vocabulary by trie decoding, reading the adapted PetBERT encoder output. Highest risk (hallucination on the Uncommon bucket, weeks of work); reserve until the cheaper ideas are exhausted.

## Smaller ideas left over from closed experiments
- **Richer label text.** Labels are embedded as `"{term} {group}"` (`ml/report_mapping/training/label_presence.py`). A richer string (adding a "type of group" phrase or the ICD-O code) is a cheap change for the backbone pairs and downstream heads, but it changes the embeddings and needs a full candidate generation.
- **More backbone epochs.** The backbone loss was still falling at epoch 3 (1.97, 1.58, 1.45); more epochs were never tested.
- **All-sections-empty fallback.** About 0.25% of cases have none of the three sections filled (the text sits in gross description or clinical abstract) and get a near-zero vector. A fallback source was proposed but not built.
- **Label-side NOS de-biasing.** Improving the many NOS predictions needs the annotation's NOS preference reduced (re-annotation) and a retrain; report-text rules cannot do it ([record](experiments/2026-05-29-nos-subtype-demotion.md)).
- **Bigger, newer temporal holdout** for recency weighting, if it is ever revisited ([record](experiments/2026-05-29-recency-weighting.md)).

## Considered and not pursued (no measurement)
- **More LLM annotation for minority groups.** Judged not viable: the cascade already runs on all available data, so another run re-annotates the same cases and creates no new ground truth. The path forward for label quality is the gold/silver plan ([icd-mapping-strategy](../concepts/icd-mapping-strategy.md)).
