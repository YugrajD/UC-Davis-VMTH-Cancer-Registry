"""Public API for the keyword annotation pipeline."""

from .pipeline import KeywordConfig, run_keyword_scan


def annotate_with_defaults(
    csv_path: str,
    labels_csv_path: str,
    out_dir: str,
    *,
    id_col: str = "case_id",
    diag_num_col: str = "diagnosis_number",
    text_col: str = "diagnosis",
    max_rows: int | None = None,
) -> None:
    """Run the keyword scan with project-level path arguments.

    Handles KeywordConfig construction internally — callers don't need
    to know the config structure.
    """
    run_keyword_scan(KeywordConfig(
        csv_path=csv_path,
        id_col=id_col,
        diag_num_col=diag_num_col,
        text_col=text_col,
        labels_csv_path=labels_csv_path,
        out_dir=out_dir,
        max_rows=max_rows,
    ))
