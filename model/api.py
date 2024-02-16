from fastapi import FastAPI

app = FastAPI()



@app.get("/webhook")
async def root():
    return {"message": "Hello World"}
