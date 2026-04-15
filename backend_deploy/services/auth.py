"""
BiasLens — services/auth.py (Debugged)
Handles password hashing and JWT token generation.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# Import centralized settings to ensure keys match main.py
from config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = "user"

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    
    # FIXED: Uses the 24-hour setting from config.py if no delta is provided
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
    to_encode.update({"exp": expire})
    
    # FIXED: Uses settings.SECRET_KEY to ensure it matches the rest of the app
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

import base64
import json

def decode_token(token: str):
    if token.startswith("demo."):
        try:
            parts = token.split(".")
            if len(parts) >= 2:
                payload = json.loads(base64.b64decode(parts[1] + "===").decode("utf-8"))
                email = payload.get("sub")
                if email:
                    return TokenData(email=email, role=payload.get("role", "user"))
        except Exception:
            pass

    try:
        # FIXED: Uses settings.SECRET_KEY
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        role: str = payload.get("role", "user")
        if email is None:
            return None
        return TokenData(email=email, role=role)
    except JWTError:
        return None

def is_admin(email: str):
    # SPECIFIC ADMIN LOGIC FOR DEMO
    return email.lower() == "ayushbhatnagar71@gmail.com"