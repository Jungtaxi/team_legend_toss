"""
Weighted Ensemble for Submission Files using DuckDB
"""
import duckdb
from pathlib import Path
from typing import Dict


def ensemble_submissions(
    weights_dict: Dict[str, float],
    submissions_dir: str,
    output_file: str,
) -> str:
    total_weight = sum(weights_dict.values())
    if abs(total_weight - 1.0) > 1e-6:
        print(f"Warning: Weights sum to {total_weight}, normalizing to 1.0")
        weights_dict = {k: v/total_weight for k, v in weights_dict.items()}

    submissions_dir = Path(submissions_dir)
    output_path = submissions_dir / output_file

    for file_name in weights_dict.keys():
        file_path = submissions_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Submission file not found: {file_path}")

    con = duckdb.connect(":memory:")

    print("Loading submission files...")

    for i, (file_name, weight) in enumerate(weights_dict.items()):
        file_path = submissions_dir / file_name
        table_name = f"sub_{i}"

        print(f"  Loading {file_name} (weight: {weight:.4f})")

        con.execute(f"""
            CREATE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{file_path}')
        """)

        result = con.execute(f"SELECT COUNT(*) as count FROM {table_name}").fetchone()
        print(f"    Rows: {result[0]:,}")

    print("\nCreating weighted ensemble...")

    file_names = list(weights_dict.keys())
    weights = list(weights_dict.values())

    weighted_sum = " + ".join([
        f"(sub_{i}.clicked * {weights[i]})"
        for i in range(len(weights))
    ])

    join_clause = "sub_0"
    for i in range(1, len(file_names)):
        join_clause += f"\n    INNER JOIN sub_{i} ON sub_0.ID = sub_{i}.ID"

    ensemble_query = f"""
        SELECT
            sub_0.ID,
            ({weighted_sum}) as clicked
        FROM {join_clause}
        ORDER BY sub_0.ID
    """

    print(f"\nExecuting ensemble query...")

    con.execute(f"""
        COPY (
            {ensemble_query}
        ) TO '{output_path}' (HEADER, DELIMITER ',')
    """)

    verify_result = con.execute(f"""
        SELECT
            COUNT(*) as count,
            MIN(clicked) as min_clicked,
            MAX(clicked) as max_clicked,
            AVG(clicked) as avg_clicked
        FROM read_csv_auto('{output_path}')
    """).fetchone()

    print(f"\nEnsemble complete!")
    print(f"  Output: {output_path}")
    print(f"  Rows: {verify_result[0]:,}")
    print(f"  Clicked range: [{verify_result[1]:.6f}, {verify_result[2]:.6f}]")
    print(f"  Clicked mean: {verify_result[3]:.6f}")

    print(f"\nWeight summary:")
    for file_name, weight in weights_dict.items():
        print(f"  {file_name}: {weight:.4f}")

    con.close()

    return str(output_path)
