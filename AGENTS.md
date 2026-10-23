# Agent rules

General, project-independent Agent development practices are documented in `skills.md`; this file contains repository-specific rules and constraints.

- For every new task, present a concrete implementation plan and wait for the user's explicit approval before making code or documentation changes. Clarifying requirements is not plan approval.
- Do not invent interfaces, parameters, data semantics, or business conditions. Ask the user when required details are unclear.
- Do not hard-code customer-question phrasing or special cases in tools. Use generic query parameters and reusable data access logic.
- Report-specific section names and source headers belong only in source adapters. Tools and business logic must use the canonical data model.
- Preserve source files and import metadata so normalized records can be traced to their origin.
- Do not inspect `data/Activity_Statement_1Y.csv` unless the user explicitly asks.
- Keep verification proportional; use smoke tests for end-to-end behavior unless the user requests more.
- Before changing intent response templates, answer boundaries, output checks, or their evaluations, read and follow `solution_design.md`. Keep templates at the semantic-contract level; do not bind them to concrete tool names, signatures, or result fields. Update the design document when the contract changes.
