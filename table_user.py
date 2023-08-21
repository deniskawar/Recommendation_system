from database import *
from sqlalchemy import Column, Integer, String, text, func, TIMESTAMP, ForeignKey
from sqlalchemy.orm import relationship


class User(Base):
    __tablename__ = 'user'
    __tableargs__ = {"schema": "cd"}
    id = Column(Integer, primary_key=True)
    gender = Column(Integer)
    age = Column(Integer)
    country = Column(String)
    city = Column(String)
    exp_group = Column(Integer)
    os = Column(String)
    source = Column(String)