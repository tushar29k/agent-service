"""HTTP service over ReActAgent. Run: uvicorn service:app --reload

  POST /run      {thread_id, message}  -> streams events (JSON lines)
  POST /approve  {thread_id, approved} -> resumes after an approval gate
  GET  /health
"""
import json

try:
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
except ImportError as e:
    raise SystemExit("pip install fastapi uvicorn  (then re-run)") from e

from agent import ReActAgent

agent = ReActAgent()
app = FastAPI(title="agent-service")


def event_stream(events):
    for ev in events:
        yield json.dumps(ev) + "\n"


@app.post("/run")
def run(body: dict):
    return StreamingResponse(
        event_stream(agent.run(body["thread_id"], body["message"])),
        media_type="application/x-ndjson")


@app.post("/approve")
def approve(body: dict):
    return StreamingResponse(
        event_stream(agent.approve(body["thread_id"], body["approved"])),
        media_type="application/x-ndjson")


@app.get("/health")
def health():
    return {"status": "ok"}
