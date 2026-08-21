# Transactions skill

Use this skill whenever the user asks about money movement, transactions, income, expenses, refunds, or loan payments.

## Filtering rules

- Type is one of 'income', 'expense', 'refund'. CategoryGroup distinguishes 'operational' vs 'direct' expenses.
- Loan payments are transactions with LoanId set. If the user asks for "loan payment history", prefer the loan-payment breakdown columns (PrincipalPaid, InterestPaid, EscrowPaid, PMIPaid, LateFees) over the total Amount.
- Account scoping, soft-delete exclusion, and split-transaction exclusion are applied automatically — never add these filters yourself.

## Answering patterns

- When reporting a total, say whether it is income or expense and over which period.
- If the user asks for "spending" or "costs", default to expenses only; ask only if genuinely ambiguous.
- When nothing matches the filters, report "no transactions found" rather than inventing rows.
