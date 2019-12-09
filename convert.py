import os
from google.cloud import storage
import tfcoreml
import coremltools
from coremltools.models.neural_network.quantization_utils import quantize_weights

# GCP Credentials taken from GOOGLE_APPLICATION_CREDENTIALS env

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

SOURCE_BUCKET = os.getenv('SOURCE_BUCKET', None)
SOURCE_MODEL_PATH = os.getenv('SOURCE_MODEL_PATH', None)
SOURCE_LABELS_PATH = os.getenv('SOURCE_LABELS_PATH', None)
DESTINATION_BUCKET = os.getenv('DESTINATION_BUCKET', None)
DESTINATION_DIRECTORY = os.getenv('DESTINATION_DIRECTORY', None)

download_blob(SOURCE_BUCKET, SOURCE_MODEL_PATH, '/tmp/model.h5')
download_blob(SOURCE_BUCKET, SOURCE_LABELS_PATH, '/tmp/labels.txt')

# Convert h5 model to coreml
OUTPUT_NAME = ['Identity']
MODEL_LABELS = '/tmp/labels.txt'
model = tfcoreml.convert('./tmp/kiosk_model.h5',
    image_input_names = ['input_1'],
    input_name_shape_dict={'input_1': (1, 224, 224, 3)},
    output_feature_names=OUTPUT_NAME,
    minimum_ios_deployment_target='13',
    red_bias = -1,
    green_bias = -1,
    blue_bias = -1,
    is_bgr = True,
    image_scale = 2.0/255.0,
)
model.save('/tmp/model.mlmodel')

# Create quantised version of coreml model
model = coremltools.models.MLModel('/tmp/model.mlmodel')
quantized_model = quantize_weights(model, nbits=8, quantization_mode="linear")
quantized_model.save('/tmp/model_quant.mlmodel')

upload_blob(DESTINATION_BUCKET, '/tmp/model.mlmodel', DESTINATION_DIRECTORY + '/model.mlmodel')
upload_blob(DESTINATION_BUCKET, '/tmp/model_quant.mlmodel', DESTINATION_DIRECTORY + '/model_quant.mlmodel')
