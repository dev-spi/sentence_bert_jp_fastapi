from typing import Optional
from fastapi import FastAPI
import json

from app.sentencebertjapanese import SentenceBertJapanese

app = FastAPI()

MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"  # <- v2です。
model = SentenceBertJapanese(MODEL_NAME)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.get("/vector/{sentence}")
def get_vec(sentence: str ):
    try:
        sentence_vector = model.sentence_to_vec(sentence)
        return {
            "statusCode": 200,
            "body": json.dumps({"sentence_vector": sentence_vector})
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            #"headers": {
            #    'Content-Type': 'application/json',
            #    'Access-Control-Allow-Origin': '*',
            #    "Access-Control-Allow-Credentials": True
            #},
            "body": json.dumps({"error": repr(e)})
        }