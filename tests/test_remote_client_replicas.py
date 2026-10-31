import itertools

import pytest
import requests

from code_executor_py import RemoteExecutor
from code_executor_py.remote_executor import serialize_data

CODE = "def double(data):\n    return data * 2\n"


class _Response:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)


class _TwoReplicaServer:
    """Round-robins requests over two pods that each keep their own compiled functions."""

    def __init__(self):
        self.pods = [set(), set()]
        self._next_pod = itertools.cycle(self.pods)
        self.execute_bodies = []

    def post(self, url, json):
        pod = next(self._next_pod)
        function_id = "fn-" + str(hash(json.get("code")))
        if url.endswith("/create_function"):
            pod.add(function_id)
            return _Response(200, {"function_id": function_id})
        self.execute_bodies.append(json)
        if json.get("code") is None and json["function_id"] not in pod:
            return _Response(404, {"detail": "Function not found"})
        return _Response(200, {"result": serialize_data("ok")})


@pytest.fixture
def server(monkeypatch):
    fake = _TwoReplicaServer()
    monkeypatch.setattr(requests, "post", fake.post)
    return fake


def test_every_call_succeeds_whichever_replica_serves_it(server):
    run = RemoteExecutor("http://executor").create_executable(CODE, "double")

    assert [run(n) for n in range(4)] == ["ok"] * 4


def test_execute_sends_the_code_and_still_the_function_id(server):
    RemoteExecutor("http://executor").create_executable(CODE, "double", {"TZ": "UTC"})(1)

    body = server.execute_bodies[0]
    assert body["function_id"].startswith("fn-")
    assert (body["code"], body["function_name"], body["env_vars"]) == (CODE, "double", {"TZ": "UTC"})
