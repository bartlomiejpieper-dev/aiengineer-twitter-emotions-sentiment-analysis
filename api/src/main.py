from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
import mypreprocessing as mp
import os

value_label_dict = {
    0: 'sadness', 
    1: 'joy', 
    2: 'love', 
    3: 'anger', 
    4: 'fear', 
    5: 'surprise'}

app = FastAPI()

model_dir = os.path.join("api", "src", "model")
tokenizer = mp.load_tokenizer(os.path.join(model_dir, "tokenizer.json"))
model = tf.keras.models.load_model(os.path.join(model_dir, "model.keras"))

class PredictionInput(BaseModel):
    features: list[str]
    
@app.get("/model_info")
async def model_info():
    try:
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))        
        formatted_summary = "```\n" + "\n".join(model_summary) + "\n```"

        model_info = {
            "input_shape": model.input_shape[1:], 
            "output_shape": model.output_shape[1:], 
            "number_of_layers": len(model.layers),
            "layer_info": [
                {
                    "name": layer.name,
                    "input_shape": layer.input_shape[1:], 
                    "output_shape": layer.output_shape[1:], 
                    "trainable": layer.trainable
                } for layer in model.layers
            ],
            "model_summary": formatted_summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model information: {e}")

    return model_info

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        print(input_data)
        features = pd.Series(input_data.features)
        features = mp.lower_case(features)
        features = mp.sanitize(features)
        features = mp.remove_stop_words(features)
        features = mp.tokenize_to_words(features)
        features = mp.stem_tokens(features)
        features = mp.tokenize_to_numbers(features, tokenizer)
        features = mp.trim_pad(features, maxlen=40)
        features = np.array(features)
        
        predicted_probs = model.predict(features)
        predicted_classes = np.argmax(predicted_probs, axis=1).tolist()
        predicted_class_names = [value_label_dict[pred_class] for pred_class in predicted_classes]
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {e}")

    result = {"predicted_class_name": predicted_class_names, "predicted_class": predicted_classes, "probabilities": predicted_probs.tolist()}    
    return result
