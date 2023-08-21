from datetime import datetime
from loguru import logger
import psycopg2
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends
from typing import List
from datetime import date, timedelta


class UserGet(BaseModel):
    id: int
    gender: int
    age: int
    country: str
    city: str
    exp_group: int
    os: str
    source: str

    class Config:
        orm_mode = True


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class FeedGet(BaseModel):
    user_id: int
    post_id: int
    action: str
    time: datetime
    user: UserGet
    post: PostGet

    class Config:
        orm_mode = True


