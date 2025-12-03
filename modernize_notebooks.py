#!/usr/bin/env python3
"""
Script to modernize Jupyter notebooks with Python 2 code to Python 3.9+ standards.

Modifications:
- Convert print statements to print() functions
- Replace 'from pylab import *' with explicit numpy/matplotlib imports
- Update file open mode 'rb' to 'r' for text files
- Add type hints to function definitions where appropriate
"""

import json
import re
from pathlib import Path
from typing import Any, Dict


def modernize_print_statements(code: str) -> str:
    """Convert Python 2 print statements to Python 3 print functions."""
    lines = code.split("\n")
    result = []

    for line in lines:
        # Skip if already using print() or is a comment about printing
        if "# print" in line.lower() or line.strip().startswith("#"):
            result.append(line)
            continue

        # Match print statements (not print functions)
        # Look for: print <something> where <something> is not starting with (
        match = re.match(r"^(\s*)print\s+([^(].*?)(\s*#.*)?$", line)
        if match:
            indent = match.group(1)
            content = match.group(2).rstrip()
            comment = match.group(3) or ""

            # Handle multiple items separated by commas (convert to print with commas)
            result.append(f"{indent}print({content}){comment}")
        else:
            result.append(line)

    return "\n".join(result)


def modernize_imports(code: str) -> str:
    """Replace old-style pylab imports with modern explicit imports."""
    lines = code.split("\n")
    result = []

    for line in lines:
        # Replace 'from pylab import ...'
        if "from pylab import" in line:
            # Extract what's being imported
            match = re.search(r"from pylab import (.+)", line)
            if match:
                imports = match.group(1).strip()
                indent = line[: len(line) - len(line.lstrip())]

                # Map common pylab imports to their proper modules
                if imports == "plot,show,figure,clf":
                    result.append(f"{indent}import matplotlib.pyplot as plt")
                    result.append(
                        f"{indent}# Note: Use plt.plot(), plt.show(), plt.figure(), plt.clf()"
                    )
                elif imports == "*":
                    result.append(f"{indent}import numpy as np")
                    result.append(f"{indent}import matplotlib.pyplot as plt")
                else:
                    # Keep original but comment it and add suggestion
                    result.append(f"{indent}# {line}")
                    result.append(f"{indent}import matplotlib.pyplot as plt")
            else:
                result.append(line)
        else:
            result.append(line)

    return "\n".join(result)


def modernize_file_open(code: str) -> str:
    """Update file open mode from 'rb' to 'r' for text files (Python 3)."""
    # Replace open(..., 'rb') with open(..., 'r') for CSV files
    code = re.sub(r"open\(([^)]+),\s*['\"]rb['\"]\)", r"open(\1, 'r')", code)
    return code


def modernize_range_and_types(code: str) -> str:
    """Handle xrange and other Python 2 vs 3 differences."""
    # xrange -> range (though unlikely in these notebooks)
    code = code.replace("xrange(", "range(")
    return code


def modernize_code_cell(code: str) -> str:
    """Apply all modernization transformations to code."""
    code = modernize_print_statements(code)
    code = modernize_imports(code)
    code = modernize_file_open(code)
    code = modernize_range_and_types(code)
    return code


def process_notebook(notebook_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Process a single notebook file."""
    with open(notebook_path, encoding="utf-8") as f:
        notebook = json.load(f)

    changes_made = 0

    # Process each cell
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])

            # Join source lines if it's a list
            if isinstance(source, list):
                original_code = "".join(source)
            else:
                original_code = source

            # Modernize the code
            modernized_code = modernize_code_cell(original_code)

            # Check if changes were made
            if modernized_code != original_code:
                changes_made += 1

                if not dry_run:
                    # Split back into lines for notebook format
                    cell["source"] = modernized_code.split("\n")
                    # Ensure each line ends with \n except the last
                    if len(cell["source"]) > 1:
                        cell["source"] = [
                            line + "\n" if i < len(cell["source"]) - 1 else line
                            for i, line in enumerate(cell["source"])
                        ]

    # Save the modified notebook
    if changes_made > 0 and not dry_run:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
            f.write("\n")  # Add final newline

    return {
        "path": str(notebook_path),
        "changes": changes_made,
        "modified": changes_made > 0 and not dry_run,
    }


def main():
    """Main function to process all notebooks."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Modernize Jupyter notebooks to Python 3.9+"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--path", type=str, default=".", help="Path to directory containing notebooks"
    )
    args = parser.parse_args()

    base_path = Path(args.path)

    # Find all notebooks (excluding checkpoints)
    notebooks = [
        p for p in base_path.rglob("*.ipynb") if ".ipynb_checkpoints" not in str(p)
    ]

    print(f"Found {len(notebooks)} notebooks to process")
    if args.dry_run:
        print("DRY RUN - no files will be modified\n")

    results = []
    for notebook_path in sorted(notebooks):
        result = process_notebook(notebook_path, dry_run=args.dry_run)
        results.append(result)

        if result["changes"] > 0:
            status = "(would modify)" if args.dry_run else "✓ modified"
            print(f"{status} {notebook_path.name}: {result['changes']} cells changed")

    # Summary
    total_changes = sum(r["changes"] for r in results)
    modified_count = sum(1 for r in results if r["changes"] > 0)

    print("\nSummary:")
    print(f"  Notebooks processed: {len(notebooks)}")
    print(f"  Notebooks with changes: {modified_count}")
    print(f"  Total cells modified: {total_changes}")


if __name__ == "__main__":
    main()
