from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf

# Define the FastAPI app
app = FastAPI()

# Define the request payload schema for user-based recommendations
class UserRequest(BaseModel):
    user_id: int

# Define the request payload schema for item-based recommendations
class ItemRequest(BaseModel):
    item_id: int

# Load your trained TensorFlow recommender model
model = tf.keras.models.load_model('path_to_your_model')

# Define the endpoint for user-based recommendations
@app.post('/recommend/user')
def user_based_recommendation(request: UserRequest):
    user_id = request.user_id

    # Preprocess the input data
    user_feature = tf.convert_to_tensor([[user_id]])

    # Perform user-based recommendation using the trained model
    user_based_prediction = model.predict([user_feature, None])

    # Return the user-based recommendation scores
    return {'user_based_recommendations': user_based_prediction.tolist()}

# Define the endpoint for item-based recommendations
@app.post('/recommend/item')
def item_based_recommendation(request: ItemRequest):
    item_id = request.item_id

    # Preprocess the input data
    item_feature = tf.convert_to_tensor([[item_id]])

    # Perform item-based recommendation using the trained model
    item_based_prediction = model.predict([None, item_feature])

    # Return the item-based recommendation scores
    return {'item_based_recommendations': item_based_prediction.tolist()}
