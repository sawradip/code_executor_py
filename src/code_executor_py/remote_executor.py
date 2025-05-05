import zlib
import pickle
import base64
import requests
from pathlib import Path
from typing import Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from .venv_executor import VenvExecutor


def serialize_data(obj, compression_level=9):
    """Convert a Python object to a compressed, base64-encoded string."""
    pickled = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zlib.compress(pickled, compression_level)
    encoded = base64.b64encode(compressed)
    return encoded.decode('ascii')


def deserialize_data(string):
    """Convert a compressed, base64-encoded string back to a Python object."""
    encoded = string.encode('ascii')
    compressed = base64.b64decode(encoded)
    pickled = zlib.decompress(compressed)
    return pickle.loads(pickled)


class CodeRequest(BaseModel):
    code: str
    function_name: Optional[str] = None


class ExecuteRequest(BaseModel):
    function_id: str
    function_params: str  # Compressed, serialized data


class ExecuteResponse(BaseModel):
    result: str  # Compressed, serialized result


class RemoteExecutorServer:
    def __init__(self,
                 host: str = "0.0.0.0", 
                 port: int = 8099,
                 venv_path: Optional[Union[str, Path]] = None,
                 base_packages: Optional[list[str]] = None,
                 llm = None):
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.venv_executor = VenvExecutor(
            venv_path=venv_path,
            base_packages=base_packages,
            llm=llm
        )
        self.functions = {}

        @self.app.post("/create_function")
        async def create_function(request: CodeRequest):
            try:
                code_hash = str(hash(request.code))

                if code_hash in self.functions:
                    return {"function_id": code_hash}

                func = self.venv_executor.create_executable(
                    request.code,
                    request.function_name
                )
                self.functions[code_hash] = func
                return {"function_id": code_hash}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/execute", response_model=ExecuteResponse)
        async def execute(request: ExecuteRequest):
            # try:
            func = self.functions.get(request.function_id)
            if not func:
                raise HTTPException(
                    status_code=404,
                    detail="Function not found"
                )

            params = deserialize_data(request.function_params)
            args, kwargs = params["function_args"], params["function_kwargs"]
            # if isinstance(params, tuple) and len(params) == 2:
            #     args, kwargs = params
            #     result = func(*args, **kwargs)
            # elif isinstance(params, dict):
            #     result = func(**params)
            # else:
            #     result = func(*params)
            result = func(*args, **kwargs)
            return ExecuteResponse(result=serialize_data(result))
            # except Exception as e:
            #     raise HTTPException(status_code=400, detail=str(e))

    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)


class RemoteExecutor:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')

    def create_executable(
        self,
        function_code: str,
        function_name: Optional[str] = None
    ) -> callable:
        """Create an executable function that runs on the remote server."""

        response = requests.post(
            f"{self.server_url}/create_function",
            json={"code": function_code, "function_name": function_name}
        )
        response.raise_for_status()
        function_id = response.json()["function_id"]

        def wrapper(*args, **kwargs):

            # if kwargs and args:
            #     params = (args, kwargs)
            # elif kwargs:
            #     params = kwargs
            # else:
            #     params = args
            serialized_params = serialize_data({
                    "function_args": args,
                    "function_kwargs": kwargs
            })

            response = requests.post(
                f"{self.server_url}/execute",
                json={
                    "function_id": function_id,
                    "function_params": serialized_params
                }
            )
            response.raise_for_status()
            return deserialize_data(response.json()["result"])

        return wrapper

