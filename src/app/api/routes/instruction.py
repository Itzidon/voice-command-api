import json

from fastapi import APIRouter, HTTPException, status
from groq import Groq

from src.app.core.config import get_settings
from src.app.schemas.voice import InstructionPayload, InstructionRequest

router = APIRouter(tags=["instruction"])


SYSTEM_PROMPT = """
You are a routing assistant for a task management API.

Your job is to convert the user's natural language instruction into ONLY
one valid JSON object.

The JSON must always have exactly this structure:

{
  "endpoint": "/tasks",
  "method": "POST",
  "params": {}
}

Allowed actions:

1. List all tasks
{
  "endpoint": "/tasks",
  "method": "GET",
  "params": {}
}

2. Create a task
{
  "endpoint": "/tasks",
  "method": "POST",
  "params": {
    "title": "task title"
  }
}

3. Replace a complete task
{
  "endpoint": "/tasks/{task_id}",
  "method": "PUT",
  "params": {
    "title": "new title",
    "done": false
  }
}

4. Partially update a task
{
  "endpoint": "/tasks/{task_id}",
  "method": "PATCH",
  "params": {
    "done": true
  }
}

or

{
  "endpoint": "/tasks/{task_id}",
  "method": "PATCH",
  "params": {
    "title": "new title"
  }
}

5. Delete a task
{
  "endpoint": "/tasks/{task_id}",
  "method": "DELETE",
  "params": {}
}

Rules:
- Respond ONLY with valid JSON.
- Never include markdown.
- Never include explanations.
- Never include text before or after the JSON.
- Use only GET, POST, PUT, PATCH or DELETE.
- Infer the action from the user's instruction.
- Do not use hardcoded keyword matching.
"""


@router.post("/instruction", response_model=InstructionPayload)
def route_instruction(
    payload: InstructionRequest,
) -> InstructionPayload:
    settings = get_settings()

    try:
        client = Groq(api_key=settings.groq_api_key)

        completion = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": payload.transcription,
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = completion.choices[0].message.content

        if not content:
            raise ValueError("Groq returned an empty response")

        data = json.loads(content)

        return InstructionPayload(
            endpoint=data["endpoint"],
            method=data["method"],
            params=data.get("params", {}),
        )

    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Groq returned invalid JSON",
        ) from error

    except (KeyError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Groq returned an invalid instruction format",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing instruction: {str(error)}",
        ) from error