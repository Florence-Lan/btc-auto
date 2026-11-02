"""Build an immutable test feed identity without touching canceled experiments."""
from copy import deepcopy
import json


def isolate_timing_feed(directory, monkeypatch, feed):
    # Exercise the real design/fingerprint implementation. Production archives
    # intentionally retain their old source pins after the user cancels them.
    # They cannot be rewritten or treated as current merely to satisfy tests.
    _, rules, fingerprint = feed.timing_trial.load_design()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'rules': deepcopy(rules), 'fingerprint': deepcopy(fingerprint)}
    (directory/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setattr(feed, 'SOURCE', directory)
    return directory
