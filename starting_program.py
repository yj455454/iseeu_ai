from keras.models import load_model

model = load_model('korean400_batch32_0.847.h5', compile=False)

print(model.summary())


