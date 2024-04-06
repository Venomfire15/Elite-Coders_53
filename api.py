from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="API_KEY"
)

result = CLIENT.infer(your_image.jpg, model_id="droneicip-0m0ux/1")