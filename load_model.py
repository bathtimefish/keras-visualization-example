from keras.models import model_from_json
from keras.utils import plot_model

model = None
with open('model.json') as f:
    model = model_from_json(f.read())
plot_model(model, to_file='model.png')
