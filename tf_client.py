import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf
import numpy as np
from llama.tokenizer import Tokenizer

GRPC_PORT = "8500"
GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 4096 * 4096 * 3  # Max LENGTH the GRPC should handle
tokenizer_path = 'tokenizer.model'

def serve_grpc(tokens):
    channel = grpc.insecure_channel(f'localhost:{GRPC_PORT}',
                                    options=[('grpc.max_receive_message_length', GRPC_MAX_RECEIVE_MESSAGE_LENGTH)])
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    grpc_request = predict_pb2.PredictRequest()
    grpc_request.model_spec.name = 'tfllama'
    grpc_request.model_spec.signature_name = 'serving_default'


    input_list = tokenizer.encode(tokens, b)  # make list

    tokenized = tokenizer.encode(sentence, bos=True, eos=False)

    if len(tokenized) < 32:
        tokenized = [tokenizer.pad_id] * (32 - len(tokenized)) + tokenized

    tokens = tf.cast(tf.constant(tokenized), tf.int64)
    grpc_request.inputs['args'].CopyFrom(tf.make_tensor_proto(tokens))

    predictions = stub.Predict(grpc_request, 10)

    # converting from tensor proto to numpy
    print(predictions.output)


if __name__ == '__main__':
    tokenizer = Tokenizer(model_path=tokenizer_path)
    serve_grpc(tokenizer, 'hello world this is han qi')