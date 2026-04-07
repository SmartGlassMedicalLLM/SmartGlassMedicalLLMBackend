"""
Pydantic models that define the shared request and response structures used
across all API endpoints.

Every request carries ``reqRefId`` / ``resRefId`` correlation identifiers so
that callers can trace requests.
"""

from pydantic import BaseModel

class Highlight(BaseModel):
  """
  A single highlighted term and the pages on which it appears.

  :param word: The highlighted word or phrase.
  :param pages: List of 1-indexed page numbers where the word was highlighted.
  """
  word: str
  pages: list[int]

class BaseRequest(BaseModel):
  """
  Common request shared by all endpoints.

  :param reqRefId: Client-assigned request correlation ID.
  :param resRefId: Client-assigned response correlation ID.
  :param prompt: The natural-language prompt or text to process.
  :param sessionId: Optional session identifier for stateful interactions.
  :param userId: Optional user identifier for analytics.
  :param docName: Optional name of the document being queried.
  :param currPage: Optional 1-indexed current page number in the viewed document.
  :param highlights: Optional list of highlighted terms and their page locations.
  """
  reqRefId: str
  resRefId: str
  prompt: str

  sessionId: str | None = None
  userId: str | None = None
  docName: str | None = None
  currPage: int | None = None
  highlights: list[Highlight] | None = None

class DocumentReference(BaseModel):
  """
  A document citation supporting a claim in an AI-generated answer.

  :param refId: Unique reference identifier (such as ``"ref-1"``).
  :param page: 1-indexed page number the quote was taken from.
  :param label: Short human-readable label (such as ``"Safety profile"``).
  :param quote: The verbatim excerpt from the document.
  :param highlightedWord: The highlighted word that scoped this reference, if any.
  :param confidence: Similarity score (0-1) between the claim and the quote.
  """
  refId: str
  page: int
  label: str
  quote: str
  highlightedWord: str
  confidence: float

class BaseResponse(BaseModel):
  """
  Standard success response envelope.

  :param reqRefId: Echoed from the request for correlation.
  :param resRefId: Echoed from the request for correlation.
  :param answer: The model's answer or generated text.
  :param references: Optional list of document citations supporting the answer.
  """
  reqRefId: str
  resRefId: str
  answer: str
  references: list[DocumentReference] | None = None

class APIError(BaseModel):
  """
  Structured error detail embedded inside :class:`ErrorResponse`.

  :param code: Machine-readable error code (such as ``"VALIDATION_ERROR"``).
  :param message: Human-readable error description.
  """
  code: str
  message: str

class ErrorResponse(BaseModel):
  """
  Standard error response returned when a request cannot be fulfilled.

  :param reqRefId: Echoed from the request (``"unknown"`` if unparseable).
  :param resRefId: Echoed from the request (``"unknown"`` if unparseable).
  :param error: Structured error detail.
  """
  reqRefId: str
  resRefId: str
  error: APIError