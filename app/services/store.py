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


SYNTHETIC_POLICY_INDEX = [
    {
        "policy_id": "eylf-belonging",
        "title": "EYLF V2.0 Belonging, Being and Becoming",
        "source": "synthetic-policy-index",
        "section": "Learning outcomes overview",
        "summary": "Synthetic index entry for EYLF-aligned learning outcomes.",
    },
    {
        "policy_id": "nqs-qa1-program",
        "title": "NQS QA1 Educational Program and Practice",
        "source": "synthetic-policy-index",
        "section": "Quality Area 1",
        "summary": "Synthetic index entry for program planning and reflective practice.",
    },
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
    draft_type: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="draft")


class PolicyIndexEntry(Base):
    __tablename__ = "policy_index"

    policy_id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    section: Mapped[str] = mapped_column(String, nullable=False)
    summary: Mapped[str] = mapped_column(String, nullable=False)


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
            self._seed_policy_index(session)
            session.commit()

    def get_class_profile(self, class_id: str) -> Optional[Dict[str, Any]]:
        with self.session_factory() as session:
            profile = session.get(ClassProfile, class_id)
            if profile is None:
                return None
            return self._class_profile_to_dict(profile)

    def search_policy_index(self, query: str) -> List[Dict[str, str]]:
        pattern = f"%{query.lower()}%"
        with self.session_factory() as session:
            entries = (
                session.execute(
                    select(PolicyIndexEntry)
                    .where(
                        PolicyIndexEntry.title.ilike(pattern)
                        | PolicyIndexEntry.section.ilike(pattern)
                        | PolicyIndexEntry.summary.ilike(pattern)
                    )
                    .order_by(PolicyIndexEntry.policy_id)
                )
                .scalars()
                .all()
            )
            return [self._policy_entry_to_dict(entry) for entry in entries]

    def save_draft(
        self,
        *,
        draft_id: str,
        draft_type: str,
        title: str,
        content: str,
        status: str = "draft",
    ) -> Dict[str, str]:
        draft = DraftRecord(
            draft_id=draft_id,
            draft_type=draft_type,
            title=title,
            content=content,
            status=status,
        )
        with self.session_factory() as session:
            session.add(draft)
            session.commit()

        return {
            "draft_id": draft_id,
            "draft_type": draft_type,
            "title": title,
            "status": status,
        }

    def count_class_profiles(self) -> int:
        with self.session_factory() as session:
            return len(session.execute(select(ClassProfile)).scalars().all())

    def _seed_class_profiles(self, session: Session) -> None:
        for profile in SYNTHETIC_CLASS_PROFILES:
            if session.get(ClassProfile, profile["class_id"]) is not None:
                continue
            session.add(ClassProfile(**profile))

    def _seed_policy_index(self, session: Session) -> None:
        for policy in SYNTHETIC_POLICY_INDEX:
            if session.get(PolicyIndexEntry, policy["policy_id"]) is not None:
                continue
            session.add(PolicyIndexEntry(**policy))

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

    def _policy_entry_to_dict(self, entry: PolicyIndexEntry) -> Dict[str, str]:
        return {
            "policy_id": entry.policy_id,
            "title": entry.title,
            "source": entry.source,
            "section": entry.section,
            "summary": entry.summary,
        }
