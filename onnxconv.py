import numpy as np
import tf2onnx
import onnx
from tensorflow.keras.models import load_model
import tensorflow as tf
onnx_model_name = 'act.onnx'

model = load_model('saved_models\small')
onnx_model,external_tensor_storage =  tf2onnx.convert.from_keras(model,opset=13)
onnx.save(onnx_model, "AcT_small_model.onnx")