# Procurement Evaluation Policy — exercise version
R1. Evaluate quotes only against the requested RFQ and supplier records. Missing RFQ/supplier details → NEEDS_INFORMATION.
R2. A supplier is ELIGIBLE only if onboarding_status=approved AND tax_verified=yes. Pending/suspended suppliers are INELIGIBLE regardless of price.
R3. Quote must meet full requested quantity, promised delivery on/before required_by, and unit price <= max_unit_budget_inr. Any failure = NON_COMPLIANT for this exercise. Partial fulfillment is not allowed unless explicitly re-authorized by human.
R4. Tax-inclusive total = quantity * unit_price * (1 + tax_percent / 100); calculate with Decimal and round to two decimals.
R5. Among compliant quotes, recommend the lowest tax-inclusive total. Ties: higher on-time delivery percentage wins, then lower defect percentage. These are deterministic rules.
R6. Awarding a contract and contacting a supplier always requires human procurement approval; the AI only recommends an award.
R7. If no compliant quote exists, output NO_COMPLIANT_BID and a request for a revised RFQ or human exception decision.
R8. Exact same RFQ, supplier, price, quantity and date under multiple quote IDs is POSSIBLE_DUPLICATE; flag for investigation, do not silently count twice.
R9. Every recommended quote must cite RFQ, supplier, quote and relevant policy clauses. Never invent compliance certificates or environmental claims.
