import os
import secrets
import srsly
import markdown
from typing import Annotated
from rich import print
from db import make_index
from pathlib import Path
from fastapi import FastAPI, Request, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordBearer
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials


data = list(srsly.read_jsonl("data/data.jsonl"))
# sort the data by image name
data = sorted(data, key=lambda x: x['image'])

app = FastAPI()
security = HTTPBasic()
app.mount("/assets", StaticFiles(directory='data'), name="assets")
templates = Jinja2Templates(directory="templates")

## set up the Chroma database
db_name = "marshallDB"
data_path = "data/data.jsonl"
if not Path(db_name).exists():
    print("[teal]Creating index...[/teal]")
    make_index(db_name, data)

def get_current_username(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
):
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = os.getenv("USERNAME", "admin").encode("utf8")
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = os.getenv("PASSWORD", "admin").encode("utf8")
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/")
def read_root(request: Annotated[str, Depends(get_current_username)]):
    return templates.TemplateResponse("index.html")

@app.get("/page/{image}")
def read_item(image: str , request: Request):
    item = next((item for item in data if item["image"] == image), None)
    if item:
        item_idx = data.index(item)
        previous = data[item_idx - 1]["image"] if item_idx > 0 else None
        item['previous'] = f"/page/{previous}" if previous else None
        next_ = data[item_idx + 1]["image"] if item_idx < len(data) - 1 else None
        item['next'] = f"/page/{next_}" if next_ else None
        item['text'] = markdown.markdown(item['text'])
        return templates.TemplateResponse(
                        "page.html",
                        {"request": request, "item": item},
                    )
    else:
        return {"image": image, "error": "Image not found"}

@app.post("/page/{image}")
async def write_item(image: Annotated[str, Depends(get_current_username)], request: Request):
    data = await request.json()
    print(image, data)
    #item = next((item for item in data if item["image"] == image), None)
    
    