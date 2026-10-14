# Agent rules

General, project-independent Agent development practices are documented in `skills.md`; this file contains repository-specific rules and constraints.

- Do not begin coding for a new task until the user has explicitly approved the proposed plan.
- Do not invent interfaces, parameters, data semantics, or business conditions. Ask the user when required details are unclear.
- Do not hard-code customer-question phrasing or special cases in tools. Use generic query parameters and reusable data access logic.
- Report-specific section names and source headers belong only in source adapters. Tools and business logic must use the canonical data model.
- Preserve source files and import metadata so normalized records can be traced to their origin.
- Do not inspect `data/Activity_Statement_1Y.csv` unless the user explicitly asks.
- Keep verification proportional; use smoke tests for end-to-end behavior unless the user requests more.
