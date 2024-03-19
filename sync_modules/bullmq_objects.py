from pydantic import BaseModel
from typing import Optional, Union, List


class Recording(BaseModel):
    session_id: str
    client_id: str
    path: str
    project_context: str
    next_index: int


class JobResult(BaseModel):

    session_id: str
    client_id: str
    status: str
    result: Optional[str | object]
    error: Optional[str]

    @classmethod
    def success(cls, req: Recording, result: any) -> "JobResult":
        return cls(session_id=req.session_id, client_id=req.client_id, status="success", result=result)

    @classmethod
    def error(cls, req: Recording, error: str) -> "JobResult":
        return cls(session_id=req.session_id, client_id=req.client_id, status="fail", error=error)

