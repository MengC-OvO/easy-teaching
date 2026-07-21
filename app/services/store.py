from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import settings


SYNTHETIC_CLASS_PROFILES = [
    {
        "class_id": "kangaroo-room",
        "name": "Kangaroo Room",
        "age_group": "3-5",
        "child_count": 18,
        "interests": ["outdoor play", "storytelling", "sensory exploration"],
        "safety_notes": ["synthetic data only", "check allergies before food play"],
    }
]


class Base(DeclarativeBase):
    pass


class ClassProfile(Base):
    __tablename__ = "class_profiles"

    class_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    age_group: Mapped[str] = mapped_column(String, nullable=False)
    child_count: Mapped[int] = mapped_column(Integer, nullable=False)
    interests: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    safety_notes: Mapped[List[str]] = mapped_column(JSON, nullable=False)


class DraftRecord(Base):
    __tablename__ = "drafts"

    draft_id: Mapped[str] = mapped_column(String, primary_key=True)
    idempotency_key: Mapped[Optional[str]] = mapped_column(String, unique=True, nullable=True)
    draft_type: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="draft")


class EduFlowStore:
    def __init__(self, database_url: Optional[str] = None) -> None:
        self.database_url = database_url or self._sqlite_url(settings.database_path)
        self.engine = create_engine(self.database_url, future=True)
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            future=True,
        )

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)
        with self.session_factory() as session:
            self._seed_class_profiles(session)
            session.commit()

    def get_class_profile(self, class_id: str) -> Optional[Dict[str, Any]]:
        with self.session_factory() as session:
            profile = session.get(ClassProfile, class_id)
            if profile is None:
                return None
            return self._class_profile_to_dict(profile)

    def save_draft(
        self,
        *,
        draft_id: str,
        draft_type: str,
        title: str,
        content: str,
        status: str = "draft",
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, str]:
        with self.session_factory() as session:
            if idempotency_key:
                existing = (
                    session.execute(
                        select(DraftRecord).where(
                            DraftRecord.idempotency_key == idempotency_key
                        )
                    )
                    .scalars()
                    .first()
                )
                if existing is not None:
                    return self._draft_to_dict(existing)

            draft = DraftRecord(
                draft_id=draft_id,
                idempotency_key=idempotency_key,
                draft_type=draft_type,
                title=title,
                content=content,
                status=status,
            )
            session.add(draft)
            session.commit()

        return self._draft_to_dict(draft)

    def count_class_profiles(self) -> int:
        with self.session_factory() as session:
            return len(session.execute(select(ClassProfile)).scalars().all())

    def _seed_class_profiles(self, session: Session) -> None:
        for profile in SYNTHETIC_CLASS_PROFILES:
            if session.get(ClassProfile, profile["class_id"]) is not None:
                continue
            session.add(ClassProfile(**profile))

    def _sqlite_url(self, database_path: str) -> str:
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path}"

    def _class_profile_to_dict(self, profile: ClassProfile) -> Dict[str, Any]:
        return {
            "class_id": profile.class_id,
            "name": profile.name,
            "age_group": profile.age_group,
            "child_count": profile.child_count,
            "interests": profile.interests,
            "safety_notes": profile.safety_notes,
        }

    def _draft_to_dict(self, draft: DraftRecord) -> Dict[str, str]:
        return {
            "draft_id": draft.draft_id,
            "draft_type": draft.draft_type,
            "title": draft.title,
            "status": draft.status,
        }
