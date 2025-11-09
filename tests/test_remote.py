import os
from fastapi.testclient import TestClient

from code_executor_py.remote_executor import RemoteExecutorServer, deserialize_data


def test_remote_scoped_env_vars(tmp_path):
    server = RemoteExecutorServer(
        host="127.0.0.1",
        port=0,
        venv_path=tmp_path / "remote_env",
        base_packages=[]
    )

    with TestClient(server.app) as client:
        func_code = """
import os


def read_secret():
    return os.environ.get("REMOTE_SECRET")
"""

        response = client.post(
            "/create_function",
            json={
                "code": func_code,
                "function_name": "read_secret",
                "env_vars": {"REMOTE_SECRET": "remote-only"}
            }
        )
        assert response.status_code == 200
        function_id = response.json()["function_id"]

        execute_response = client.post(
            "/execute",
            json={
                "function_id": function_id
            }
        )
        assert execute_response.status_code == 200

        result = deserialize_data(execute_response.json()["result"])
        assert result == "remote-only"

    assert os.environ.get("REMOTE_SECRET") is None


def test_remote_env_vars_isolated_between_functions(tmp_path):
    server = RemoteExecutorServer(
        host="127.0.0.1",
        port=0,
        venv_path=tmp_path / "remote_env_isolation",
        base_packages=[]
    )

    with TestClient(server.app) as client:
        func_code = """
import os


def read_secret():
    return os.environ.get("REMOTE_SHARED_SECRET")
"""

        first_response = client.post(
            "/create_function",
            json={
                "code": func_code,
                "function_name": "read_secret",
                "env_vars": {"REMOTE_SHARED_SECRET": "first"}
            }
        )
        assert first_response.status_code == 200
        first_id = first_response.json()["function_id"]

        second_response = client.post(
            "/create_function",
            json={
                "code": func_code,
                "function_name": "read_secret",
                "env_vars": {"REMOTE_SHARED_SECRET": "second"}
            }
        )
        assert second_response.status_code == 200
        second_id = second_response.json()["function_id"]
        
        # Verify that different env_vars produce different function IDs
        assert first_id != second_id, f"Function IDs should be different but both are {first_id}"

        first_execute = client.post(
            "/execute",
            json={"function_id": first_id}
        )
        assert first_execute.status_code == 200
        assert deserialize_data(first_execute.json()["result"]) == "first"

        second_execute = client.post(
            "/execute",
            json={"function_id": second_id}
        )
        assert second_execute.status_code == 200
        assert deserialize_data(second_execute.json()["result"]) == "second"

    assert os.environ.get("REMOTE_SHARED_SECRET") is None
