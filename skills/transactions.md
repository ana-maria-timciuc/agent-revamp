# Transactions skill

Use this skill whenever the user asks about money movement, transactions, income, expenses, refunds, or loan payments.

## Filtering rules

- Every query must scope to the conversation's `account_id` and exclude soft-deleted rows.
- Split transactions are child rows linked via `split_transactions_id`. Always exclude children from totals and lists by adding `AND t.split_transactions_id IS NULL` — otherwise amounts get double-counted.
- `flow_type` is one of `income`, `expense`, `refund`. `flow_category` distinguishes direct vs operational expenses.
- Loan payments are transactions with `loan_id` set. If the user asks "loan payment history", prefer the loan payment columns: `principle_payment`, `interest_payment`, `escrow_payment`, `pmi_payment`, `fees_late_charges`.

## Answering patterns

- When reporting a total, say whether it is income or expense and over which period.
- If the user asks for "spending" or "costs", default to expenses only; ask only if genuinely ambiguous.
- When nothing matches the filters, report "no transactions found" rather than inventing rows.
