from pydantic import BaseModel
from fastapi import UploadFile

class Highlight(BaseModel):
  word: str
  pages: list[int]

class BaseRequest(BaseModel):
  reqRefId: str
  resRefId: str
  prompt: str

  sessionId: str | None = None
  userId: str | None = None
  docName: str | None = None
  currPage: int | None = None
  highlights: list[Highlight] | None = None
  pdf: UploadFile | None = None
