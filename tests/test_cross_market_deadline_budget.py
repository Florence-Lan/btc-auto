"""Worker scheduling delay cannot renew a multi-request source's deadline."""
from types import SimpleNamespace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cross_market_observations as observations
from test_macro_maintenance_integration import euro_source,archive,brief,RECEIVED


def test_worker_start_delay_consumes_the_original_source_budget(monkeypatch):
    clock=SimpleNamespace(value=0.)
    clock.monotonic=lambda:clock.value
    monkeypatch.setattr(observations,'time',clock)
    class DelayedWorker:
        def run(self,key,callback):
            clock.value=.8
            return callback()
    monkeypatch.setattr(observations,'_DEADLINES',{1.:DelayedWorker()})
    source=euro_source();source['timeout_seconds']=1.
    calls=[]
    def fetch(spec,remaining):
        calls.append((spec['url'],remaining))
        clock.value+=.3
        return archive() if spec['url']==source['url'] else brief()
    _,_,_,error=observations._source_fetch(source,fetch,lambda:RECEIVED)
    assert len(calls)==1
    assert 0<calls[0][1]<=.2
    assert error['category']=='timeout'
