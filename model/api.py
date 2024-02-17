from fastapi import FastAPI

app = FastAPI()



@app.get("/webhook")
async def root():
    return {
        "This is a local endpoint to recieve webhook payloads from github. Ngrok is used for tunneling."
    }
