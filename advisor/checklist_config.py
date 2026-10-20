"""Global answer-check definitions and per-intent check lists."""

GLOBAL_CHECKS = {
    "numeric_provenance": {
        "enabled": True,
        "failure_note": "数字校验未通过，部分数字无法在工具结果中核实，请谨慎使用。",
    },
    "product_policy": {
        "enabled": True,
        "failure_note": "这条回答未能通过产品问答边界校验，请谨慎使用。",
    },
}

GLOBAL_CHECKLIST = ["numeric_provenance"]

INTENT_CHECKLISTS = {
    "data_query": [],
    "product_info": ["product_policy"],
    "unsupported": [],
}
