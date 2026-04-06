import importlib
import pkgutil
import endpoints
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from utils.req_res_structures import ErrorResponse, APIError

app = FastAPI()

# Register custom exception/error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    data_response = ErrorResponse(
        reqRefId="unknown",
        resRefId="unknown",
        error=APIError(
            code="VALIDATION_ERROR",
            message="\n".join([
                ":".join(str(s) for s in err["loc"]) + "|" + err["msg"]
                for err in exc.errors()
            ])
        )
    )
    return JSONResponse(
        status_code=422,
        content=data_response.model_dump(),
    )

# Load endpoints from /endpoints
for _, module_name, _ in pkgutil.iter_modules(endpoints.__path__):
    module = importlib.import_module(f"endpoints.{module_name}")
    if hasattr(module, "router"):
        app.include_router(module.router)

# Run the webserver if this script is main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
