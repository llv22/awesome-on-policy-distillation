#!/usr/bin/env bash
# Recompute the entry count badge in README.md.
# Counts list entries (`- [Title](url) — desc`) plus table data rows in the
# (H2, H3) section pairs registered as entry tables. Excludes the TOC under
# `## Contents`, the Taxonomy cross-cutting tables, and any other tables.
# Idempotent: no write if the count is already current.
#
# Format invariants (uncounted if violated):
#   - List entries are single-line, use inline `[title](url)` links, and
#     contain an em-dash (—) separating title metadata from description.
#   - Entry-table header cells contain no markdown links; data cells do.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
README="$ROOT/README.md"

python3 - "$README" <<'PY'
import re, sys

path = sys.argv[1]
src = open(path).read()

# (H2, H3) section pairs whose tables count as entries. Use None for H3
# to count every table under that H2.
ENTRY_TABLE_KEYS = {
    ("Technical Reports and Industrial Recipes", None),
    ("Frameworks and Implementations", "Training Frameworks"),
}

h2 = ""
h3 = ""
count = 0

for line in src.splitlines():
    m2 = re.match(r'^##\s+(.+?)\s*$', line)
    if m2:
        h2, h3 = m2.group(1), ""
        continue
    m3 = re.match(r'^###\s+(.+?)\s*$', line)
    if m3:
        h3 = m3.group(1)
        continue

    if h2 == "Contents":
        continue

    if re.match(r'^- \[.+\]\(.+\).*—', line):
        count += 1
        continue

    if line.startswith("|") and "---" not in line and "](" in line:
        if (h2, None) in ENTRY_TABLE_KEYS or (h2, h3) in ENTRY_TABLE_KEYS:
            count += 1

pattern = re.compile(
    r'<!-- entry-count-start -->.*?<!-- entry-count-end -->',
    re.DOTALL,
)
badge = (
    f'<!-- entry-count-start --><a href="#contents">'
    f'<img src="https://img.shields.io/badge/Entries-{count}-000000'
    f'?style=for-the-badge&labelColor=000000" alt="Entries"></a>'
    f'<!-- entry-count-end -->'
)
new = pattern.sub(badge, src)
if new != src:
    open(path, 'w').write(new)
    print(f'Updated entry count to {count}')
else:
    print(f'Entry count already current ({count})')
PY
