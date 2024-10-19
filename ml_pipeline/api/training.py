from fastapi import APIRouter, HTTPException
from starlette.routing import Router

#from core.training import train_model
from  utils.save_model import  save_model
import os

router = APIRouter()

@router.post("/train")

def train():
    try:
        #train the model
        model = model_train()

        #save the model
        save_model(model, os.path.join("ml_pipeline","models","model.pkl"))
        return {"message":"Model trained and saved succesfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))







