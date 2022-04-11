import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from src.configurations import DBConfigurations

engine = create_engine(
    DBConfigurations.sql_alchemy_database_url,
    encoding="utf-8",
    pool_recycle=3600,
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base를 상속하여 Table을 만든다.
Base = declarative_base()
 

def get_db():
    db = SessionLocal()
    try:
        yield db
    except:
        db.rollback()  # 에러가 나면 롤백
        raise
    finally:
        db.close()  # 종료 전에 항상 session을 닫아준다.


# @contextmanager: with문을 사용할 수 있게 만들어줌
@contextmanager
def get_context_db():
    db = SessionLocal()
    try:
        yield db
    except:
        db.rollback()
        raise
    finally:
        db.close()
