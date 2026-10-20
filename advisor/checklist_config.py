"""Global answer-check definitions and per-intent check lists."""


class CheckDefinition(dict):
    """Check metadata with separate notes for a failed check and an unrunnable check.

    ``failure_note`` marks an answer that was reviewed and found non-compliant.
    ``unavailable_note`` marks a check that could not run at all (missing
    configuration or an unreachable reviewer), which must not be presented as if
    the answer itself were defective.
    """

    def __init__(self, enabled: bool, failure_note: str, unavailable_note: str):
        super().__init__(enabled=enabled, failure_note=failure_note, unavailable_note=unavailable_note)


GLOBAL_CHECKS: dict[str, CheckDefinition] = {
    "numeric_provenance": CheckDefinition(
        enabled=True,
        failure_note="数字校验未通过，部分数字无法在工具结果中核实，请谨慎使用。",
        unavailable_note="数字校验未能执行，数字准确性无法确认，请谨慎使用。",
    ),
    "product_policy": CheckDefinition(
        enabled=True,
        failure_note="这条回答未能通过产品问答边界校验，请谨慎使用。",
        unavailable_note="产品问答边界校验未能执行，边界合规性无法确认，请谨慎使用。",
    ),
}

GLOBAL_CHECKLIST = ["numeric_provenance"]

INTENT_CHECKLISTS = {
    "data_query": [],
    "product_info": ["product_policy"],
    "unsupported": [],
}
