from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from fastapi import FastAPI, Depends, HTTPException, status, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
import enum
import pathlib
import textwrap
from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


# Sqlite intitialization

DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI()
import google.generativeai as genai
# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Configure the Gemini API

# Create the Gemini model with custom generation settings
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=["You are a teacher giving only hints to the answer","your mission is to provide socratic questioning and visualization techniques"]
)

# Function to query the Gemini model and process the response
def generate_content(prompt: str) -> str:
    try:
        response = model.generate_content([prompt])
        print(response)  # Inspect the response object

        # Attempt to access the text content based on response structure
        if hasattr(response, 'text'):
            return response.text

        if hasattr(response, 'outputs') and isinstance(response.outputs, list):
            return response.outputs[0].text if response.outputs else "No response generated."

        return "No valid response structure found."
    except Exception as e:
        return f"Error: {str(e)}"# Function to query the Gemini model and process the response

# Home page
@app.get("/chatbot", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Tutor page
@app.get("/tutor", response_class=HTMLResponse)
async def get_tutor_page(request: Request):
    return templates.TemplateResponse("tutor.html", {"request": request})

# Endpoint to generate and return response
@app.post("/generate_response", response_class=HTMLResponse)
async def post_generate_response(request: Request, prompt: str = Form(...)):
    response = generate_content(prompt)
    return templates.TemplateResponse("tutor.html", {"request": request, "response": response})


# Enum for user roles
class UserRole(enum.Enum):
    admin = "admin"
    user = "user"

# User model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(Enum(UserRole), default=UserRole.user)

# Create the database tables
Base.metadata.create_all(bind=engine)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

# Create a default admin user if no users exist
def create_default_user(db: Session):
    if db.query(User).count() == 0:
        default_username = "admin"
        default_password = "admin"
        default_role = UserRole.admin
        hashed_password = get_password_hash(default_password)
        user = User(username=default_username, hashed_password=hashed_password, role=default_role)
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"Default admin user created: {default_username}")

# Serve the login page
@app.get("/", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# Login endpoint for form submission
@app.post("/token")
def login_for_access_token(
    db: Session = Depends(get_db),
    username: str = Form(...),
    password: str = Form(...)
):
    user = authenticate_user(db, username, password)
    if not user:
        return templates.TemplateResponse("login.html", {"request": Request, "error": "Incorrect username or password"}, status_code=status.HTTP_401_UNAUTHORIZED)
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="user_id", value=user.id)
    return response

# Dashboard route
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("dashboard.html", {"request": request, "username": user.username, "role": user.role.value})

# Dependency to create default user at startup
@app.on_event("startup")
def on_startup():
    db = SessionLocal()
    create_default_user(db)
    db.close()
