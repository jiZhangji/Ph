def tabulate(tabular_data, headers=(), tablefmt=None, floatfmt=None, **kwargs):
    """Small local fallback for the external tabulate package.

    Dassl only uses tabulate for readable dataset/config summaries.  This
    fallback keeps offline runs working when the cluster cannot reach PyPI.
    """
    rows = [list(row) for row in tabular_data]
    if headers:
        rows = [list(headers)] + rows
    if not rows:
        return ""

    widths = []
    for col_idx in range(max(len(row) for row in rows)):
        widths.append(
            max(len(str(row[col_idx])) if col_idx < len(row) else 0 for row in rows)
        )

    lines = []
    for idx, row in enumerate(rows):
        padded = [
            str(row[col_idx]).ljust(widths[col_idx]) if col_idx < len(row) else " " * widths[col_idx]
            for col_idx in range(len(widths))
        ]
        lines.append("  ".join(padded).rstrip())
        if headers and idx == 0:
            lines.append("  ".join("-" * width for width in widths).rstrip())
    return "\n".join(lines)
