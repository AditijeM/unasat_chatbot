from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    text: str = Field(..., max_length=500, description="Het bericht van de gebruiker, max 500 karakters")
    username: str = "student1"

class LoginData(BaseModel):
    username: str
    password: str

class FeedbackData(BaseModel):
    id: int
    score: str   # 'good' of 'bad'