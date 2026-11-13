"""Assign an administrator-provisioned educator account to a class."""

import argparse
from pathlib import Path
import sys
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import settings
from app.services.models import CentreRecord, ClassRecord, TeacherRecord, TeacherClassMembershipRecord


def provision(db, args):
    teacher_id = str(UUID(args.user_id))
    centre = db.get(CentreRecord, args.centre_id)
    if centre is None:
        db.add(CentreRecord(centre_id=args.centre_id, name=args.centre_name,
                            state=args.state, suburb=args.suburb))
        db.flush()
    elif not centre.active:
        raise ValueError("Centre is inactive; review its status before provisioning")
    for model, key in ((ClassRecord, args.class_id), (TeacherRecord, teacher_id)):
        existing = db.get(model, key)
        if existing and (existing.centre_id != args.centre_id or not existing.active):
            raise ValueError("Existing class or educator belongs to another centre or is inactive")
    if db.get(ClassRecord, args.class_id) is None:
        db.add(ClassRecord(class_id=args.class_id, centre_id=args.centre_id,
                           name=args.class_name, age_group=args.age_group))
    if db.get(TeacherRecord, teacher_id) is None:
        db.add(TeacherRecord(teacher_id=teacher_id, auth_user_id=teacher_id,
                             centre_id=args.centre_id, display_name=args.teacher_name))
    db.flush()
    membership = db.scalar(select(TeacherClassMembershipRecord).where(
        TeacherClassMembershipRecord.teacher_id == teacher_id,
        TeacherClassMembershipRecord.class_id == args.class_id))
    if membership is None:
        db.add(TeacherClassMembershipRecord(membership_id=str(uuid4()),
                                            teacher_id=teacher_id, class_id=args.class_id))
    elif not membership.active:
        raise ValueError("Membership is inactive; review the previous revocation before reassigning")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("user-id", "teacher-name", "centre-id", "centre-name", "state", "suburb", "class-id", "class-name", "age-group"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    engine = create_engine(settings.database_url)
    try:
        with Session(engine) as db, db.begin():
            provision(db, args)
        print("Workspace membership provisioned.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
