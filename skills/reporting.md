# Reporting skill

Use this skill when the user asks for a report, a chart, a summary of figures, or a recurring overview of their business.

## When to generate a report

- Use `generate_report` for anything the user might want to view as a chart or export, not just raw numbers.
- For simple lookups and quick numbers, answer directly from `execute_query` results.

## Presenting numbers

- Money figures: thousands separators, two decimals, and a `$` sign; negatives as `-$1,234.56`.
- When you return several figures, group them by category and mention the period covered.
- If a report request needs a date range and none is given, use the current year to date unless the user says otherwise.
