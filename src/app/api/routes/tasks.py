from fastapi import APIRouter, HTTPException, status

from src.app.schemas.voice import Task, TaskCreate, TaskReplace, TaskUpdate

router = APIRouter(prefix="/tasks", tags=["tasks"])

tasks: list[Task] = []


@router.get("", response_model=list[Task])
def get_tasks() -> list[Task]:
    return tasks


@router.post("", response_model=Task, status_code=status.HTTP_201_CREATED)
def create_task(payload: TaskCreate) -> Task:
    next_id = max((task.id for task in tasks), default=0) + 1

    new_task = Task(
        id=next_id,
        title=payload.title,
        done=payload.done,
    )

    tasks.append(new_task)
    return new_task


@router.put("/{task_id}", response_model=Task)
def replace_task(
    task_id: int,
    payload: TaskReplace,
) -> Task:
    for index, task in enumerate(tasks):
        if task.id == task_id:
            updated_task = Task(
                id=task_id,
                title=payload.title,
                done=payload.done,
            )
            tasks[index] = updated_task
            return updated_task

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Task not found",
    )


@router.patch("/{task_id}", response_model=Task)
def update_task(
    task_id: int,
    payload: TaskUpdate,
) -> Task:
    for index, task in enumerate(tasks):
        if task.id == task_id:
            updated_task = Task(
                id=task.id,
                title=payload.title if payload.title is not None else task.title,
                done=payload.done if payload.done is not None else task.done,
            )
            tasks[index] = updated_task
            return updated_task

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Task not found",
    )


@router.delete("/{task_id}")
def delete_task(task_id: int) -> dict[str, str]:
    for index, task in enumerate(tasks):
        if task.id == task_id:
            tasks.pop(index)
            return {"message": "Task deleted successfully"}

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Task not found",
    )