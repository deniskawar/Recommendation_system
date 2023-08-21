from database import *
from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey
from sqlalchemy.orm import relationship
from table_user import User
from table_post import Post


class Feed(Base):
    __tablename__ = "feed_action"
    __tableargs__ = {"schema": "cd"}
    user_id = Column(Integer, ForeignKey("user.id"), primary_key=True)
    post_id = Column(Integer, ForeignKey("post.id"), primary_key=True)
    action = Column(String)
    time = Column(TIMESTAMP)
    user = relationship("User")
    post = relationship("Post")