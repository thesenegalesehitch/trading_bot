from typing import List, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from pydantic import BaseModel

from quantum.infrastructure.db.session import get_db
from quantum.infrastructure.db.models import Course, Lesson, Quiz, Question, Option, UserProgress, User
from quantum.infrastructure.api.core.deps import get_current_user

router = APIRouter()

# --- Schémas ---
class OptionSchema(BaseModel):
    id: int
    text: str
    class Config: from_attributes = True

class QuestionSchema(BaseModel):
    id: int
    text: str
    options: List[OptionSchema]
    class Config: from_attributes = True

class LessonSchema(BaseModel):
    id: int
    title: str
    content: str
    duration: str
    class Config: from_attributes = True

class CourseSchema(BaseModel):
    id: int
    title: str
    description: str
    level: str
    image_url: str = None
    class Config: from_attributes = True

@router.get("/courses", response_model=List[CourseSchema])
async def get_courses(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Course).order_by(Course.order))
    return result.scalars().all()

@router.get("/courses/{course_id}/lessons", response_model=List[LessonSchema])
async def get_lessons(course_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Lesson).filter(Lesson.course_id == course_id).order_by(Lesson.order))
    return result.scalars().all()

@router.get("/lessons/{lesson_id}", response_model=LessonSchema)
async def get_lesson(lesson_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Lesson).filter(Lesson.id == lesson_id))
    lesson = result.scalar_one_or_none()
    if not lesson:
        raise HTTPException(status_code=404, detail="Leçon introuvable")
    return lesson

@router.get("/courses/{course_id}/quiz", response_model=Any)
async def get_quiz(course_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Quiz)
        .options(selectinload(Quiz.questions).selectinload(Question.options))
        .filter(Quiz.course_id == course_id)
    )
    quiz = result.scalar_one_or_none()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quizz introuvable pour ce cours")
    return quiz

@router.post("/lessons/{lesson_id}/complete")
async def complete_lesson(
    lesson_id: int, 
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    progress = UserProgress(user_id=current_user.id, lesson_id=lesson_id, completed=True)
    db.add(progress)
    await db.commit()
    return {"status": "success"}
