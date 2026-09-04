from datetime import datetime
from sqlalchemy import select, insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import User, Inference


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, user_id: int) -> User | None:
        result = await self.session.execute(
            select(User).where(User.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, user_name: str) -> User | None:
        result = await self.session.execute(
            select(User).where(User.username == user_name)
        )
        return result.scalar_one_or_none()


class InferenceRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def search(
        self,
        inference_id: int | None = None,
        user_id: int | None = None,
        model_id: tuple[str, str] | None = None,
        predicted_category: str | None = None,
        queried_before: datetime | None = None,
    ) -> list[Inference]:
        query = select(Inference)
        if inference_id is not None:
            query = query.where(Inference.inference_id == inference_id)
        if user_id is not None:
            query = query.where(Inference.user_id == user_id)
        if model_id is not None:
            query = query.where(Inference.model_name == model_id[0])
            query = query.where(Inference.model_version == model_id[1])
        if predicted_category is not None:
            query = query.where(
                Inference.predicted_category == predicted_category
            )
        if queried_before is not None:
            query = query.where(Inference.queried_at < queried_before)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def create(self, inference: Inference) -> Inference:
        self.session.add(inference)
        await self.session.commit()
        await self.session.refresh(inference)
        return inference

    async def create_batch(
        self,
        inferences: list[Inference]
    ) -> list[Inference]:
        values = [
            {
                "user_id": inference.user_id,
                "query_id": inference.query_id,
                "batch": inference.batch,
                "model_name": inference.model_name,
                "model_version": inference.model_version,
                "designation": inference.designation,
                "description": inference.description,
                "predicted_category": inference.predicted_category,
                "labeled_category": inference.labeled_category,
                "confidence": inference.confidence,
                "queried_at": inference.queried_at
            }
            for inference in inferences
        ]
        await self.session.execute(
            insert(Inference),
            values
        )
        await self.session.commit()
        return inferences

    async def update_label(
        self,
        inference_id: int,
        user_id: int,
        labeled_category: str
    ) -> Inference | None:
        """
        Set the confirmed/corrected category for a prediction.

        Only the user who made the original prediction can label it -- returns
        None (not an exception) if the inference doesn't exist or belongs to
        someone else, so the caller can turn that into a clean 404 without
        leaking whether the id exists at all.
        """
        result = await self.session.execute(
            select(Inference).where(Inference.inference_id == inference_id)
        )
        inference = result.scalar_one_or_none()
        if inference is None or inference.user_id != user_id:
            return None
        inference.labeled_category = labeled_category
        await self.session.commit()
        await self.session.refresh(inference)
        return inference
