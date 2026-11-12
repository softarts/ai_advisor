import json
import logging

from sa_advisor.eval.runner import EventCollector


def _record(message):
    return logging.LogRecord("sa_eval", logging.INFO, "", 0, message, (), None)


def test_event_collector_preserves_dict_log_messages():
    collector = EventCollector()
    event = {"event": "router_classified", "intent": "data_query"}

    collector.emit(_record(event))

    assert collector.events == [event]


def test_event_collector_parses_json_string_log_messages():
    collector = EventCollector()
    event = {"event": "router_fallback_unsupported", "attempt": 2}

    collector.emit(_record(json.dumps(event)))

    assert collector.events == [event]
