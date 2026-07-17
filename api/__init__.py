"""Local REST API exposing the Decision Engine for manual testing (Postman,
curl, or any HTTP client) -- not part of the analytical platform itself.

Run with:

    .venv\\Scripts\\python.exe -m uvicorn api.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger UI, or import
http://127.0.0.1:8000/openapi.json directly into Postman.
"""
