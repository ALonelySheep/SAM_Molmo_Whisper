import requests
import base64
import json
import argparse
import time


def encode_image_to_base64(image_path):
    """Encode an image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def send_request(server_url, prompt, image_path):
    """Send a request to the server with prompt and image"""
    # Encode the image
    image_base64 = encode_image_to_base64(image_path)

    # Prepare the payload
    payload = {
        "prompt": prompt,
        "image": image_base64
    }

    # Send the request
    print(f"Sending request to {server_url} with prompt: '{prompt}'")
    start_time = time.time()
    response = requests.post(
        server_url,
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    end_time = time.time()

    # Print results
    print(f"Response time: {end_time - start_time:.2f} seconds")
    print(f"Status code: {response.status_code}")

    if response.status_code == 200:
        print("Response content:")
        pretty_json = json.dumps(response.json(), indent=2)
        print(pretty_json)
        return response.json()
    else:
        print(f"Error: {response.text}")
        return None


def check_health(server_url):
    """Check if the server is running and healthy"""
    try:
        health_url = f"{server_url.rstrip('/')}/health"
        response = requests.get(health_url)
        if response.status_code == 200:
            print(f"Server is healthy: {response.json()}")
            return True
        else:
            print(f"Server returned status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to server: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test client for VLM server')
    parser.add_argument('--server', default='http://localhost:7777/process',
                        help='URL of the server endpoint')
    parser.add_argument('--prompt', default='Describe this image',
                        help='Text prompt to send')
    parser.add_argument('--image', default='../demo_data/image_1.jpg',
                        help='Path to image file')
    parser.add_argument('--health-check', action='store_true',
                        help='Only perform health check')

    args = parser.parse_args()

    if args.health_check:
        server_base = '/'.join(args.server.split('/')[:-1])
        check_health(server_base)
    else:
        if check_health('/'.join(args.server.split('/')[:-1])):
            send_request(args.server, args.prompt, args.image)
