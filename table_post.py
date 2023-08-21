from database import *
from sqlalchemy import Column, Integer, String, text, func, TIMESTAMP, ForeignKey
from sqlalchemy.orm import relationship


class Post(Base):
    __tablename__ = 'post'
    __tableargs__ = {"schema": "cd"}
    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)



if __name__ == "__main__":
    session = SessionLocal()

    #print([post.id for post in session.query(Post).filter(Post.topic == "business").order_by(text('post.id desc')).limit(10).all()])
    #result = session.query(User.country, User.os, func.count()).filter(User.exp_group == 3)\
    #   .group_by(User.country, User.os).having(func.count() > 100).order_by(func.count().desc()).all()

