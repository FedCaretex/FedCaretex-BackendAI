import secrets
import uuid
from typing import List

from sqlalchemy import Column, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class UserModel(Base):
    __tablename__ = "users"
    user_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(
        String, nullable=False
    )  # Harusnya hash, tapi demi kemudahan demo kita simpan string


class TokenModel(Base):
    __tablename__ = "tokens"
    token = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)


class HistoryModel(Base):
    __tablename__ = "history"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), index=True, nullable=False)
    timestamp = Column(Float, nullable=False)
    category = Column(String, nullable=False)
    risk_score = Column(Float, nullable=False)
    probability = Column(Float, nullable=False)
    top_feature = Column(String, nullable=False)
    model_version = Column(Integer, nullable=False)


class SqliteStore:
    def __init__(self, db_url="sqlite:///data/fedcaretex.db"):
        self.engine = create_engine(db_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._seed_demo_user()

    def _seed_demo_user(self) -> None:
        with self.SessionLocal() as db:
            user = db.query(UserModel).filter(UserModel.email == "demo@fedcaretex.ai").first()
            if not user:
                demo_user = UserModel(
                    user_id="demo-user",
                    name="Rownok",
                    email="demo@fedcaretex.ai",
                    password="demo123",
                )
                db.add(demo_user)
                db.commit()

    def register(self, name: str, email: str, password: str) -> dict:
        key = email.lower().strip()
        with self.SessionLocal() as db:
            if db.query(UserModel).filter(UserModel.email == key).first():
                raise ValueError("Email sudah terdaftar")

            user_id = str(uuid.uuid4())
            new_user = UserModel(user_id=user_id, name=name.strip(), email=key, password=password)
            db.add(new_user)
            db.commit()

            return {
                "user_id": user_id,
                "name": new_user.name,
                "email": new_user.email,
                "password": new_user.password,
            }

    def login(self, email: str, password: str) -> dict:
        with self.SessionLocal() as db:
            user = db.query(UserModel).filter(UserModel.email == email.lower().strip()).first()
            if not user or user.password != password:
                raise ValueError("Email atau kata sandi salah")

            token = secrets.token_hex(24)
            db.add(TokenModel(token=token, user_id=user.user_id))
            db.commit()

            return {
                "user_id": user.user_id,
                "name": user.name,
                "email": user.email,
                "password": user.password,
                "access_token": token,
            }

    def add_history(self, item: dict) -> None:
        with self.SessionLocal() as db:
            hist = HistoryModel(
                id=item["id"],
                user_id=item["user_id"],
                timestamp=item["timestamp"],
                category=item["category"],
                risk_score=item["risk_score"],
                probability=item["probability"],
                top_feature=item["top_feature"],
                model_version=item["model_version"],
            )
            db.add(hist)
            db.commit()

    def get_history(self, user_id: str) -> List[dict]:
        with self.SessionLocal() as db:
            records = (
                db.query(HistoryModel)
                .filter(HistoryModel.user_id == user_id)
                .order_by(HistoryModel.timestamp.desc())
                .all()
            )
            return [
                {
                    "id": r.id,
                    "user_id": r.user_id,
                    "timestamp": r.timestamp,
                    "category": r.category,
                    "risk_score": r.risk_score,
                    "probability": r.probability,
                    "top_feature": r.top_feature,
                    "model_version": r.model_version,
                }
                for r in records
            ]


store = SqliteStore()
