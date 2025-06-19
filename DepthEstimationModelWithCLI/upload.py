import base64
import os

import requests
from dotenv import load_dotenv


def get_api_key():
    """Load the API key from the .env file."""
    load_dotenv()
    api_key = os.getenv("IMG_API_KEY")
    if not api_key:
        raise ValueError("IMG_API_KEY not found in .env file")
    return api_key


def upload_image(image_path):
    """
    Upload an image to the ImgBB service.

    Args:
        image_path (str): The path to the image file to upload.

    Returns:
        str: The URL of the uploaded image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    api_key = get_api_key()
    url = "https://api.imgbb.com/1/upload"

    # METHOD 1: Base64 encoding (ImgBB'nin tercih ettiği yöntem)
    try:
        with open(image_path, "rb") as image_file:
            # Dosyayı base64'e encode et
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        # Form data olarak gönder
        payload = {
            "key": api_key,
            "image": image_data,
        }

        response = requests.post(url, data=payload)

        # HTTP status kontrolü
        if response.status_code != 200:
            raise Exception(f"HTTP Error {response.status_code}: {response.text}")

        response_data = response.json()

        # Success kontrolü
        if response_data.get("success", False):
            return response_data["data"]["url"]
        else:
            # Hata mesajını al
            if "error" in response_data:
                error_msg = response_data["error"].get("message", "Unknown error")
                error_code = response_data["error"].get("code", "Unknown code")
                raise Exception(f"ImgBB API Error [{error_code}]: {error_msg}")
            else:
                raise Exception(f"Upload failed: {response_data}")

    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")
    except Exception as e:
        # Eğer base64 method çalışmazsa, multipart dene
        return upload_image_multipart_fallback(image_path)


def upload_image_multipart_fallback(image_path):
    """Fallback method using multipart form data"""
    api_key = get_api_key()
    url = "https://api.imgbb.com/1/upload"

    try:
        with open(image_path, "rb") as image_file:
            # Files parameter ile gönder (multipart/form-data)
            files = {"image": (os.path.basename(image_path), image_file, "image/jpeg")}
            data = {"key": api_key}

            response = requests.post(url, files=files, data=data)

            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")

            response_data = response.json()

            if response_data.get("success", False):
                return response_data["data"]["url"]
            else:
                if "error" in response_data:
                    error_msg = response_data["error"].get("message", "Unknown error")
                    raise Exception(f"ImgBB API Error: {error_msg}")
                else:
                    raise Exception(f"Upload failed: {response_data}")

    except Exception as e:
        raise Exception(f"Both upload methods failed: {str(e)}")


# Test fonksiyonu
def test_upload_with_detailed_info(image_path):
    """Test upload with detailed debugging"""
    print(f"Testing upload for: {image_path}")
    print(f"File exists: {os.path.exists(image_path)}")

    if os.path.exists(image_path):
        print(f"File size: {os.path.getsize(image_path)} bytes")

    try:
        api_key = get_api_key()
        print(f"API Key loaded: {'Yes' if api_key else 'No'}")
        print(f"API Key length: {len(api_key) if api_key else 0}")

        result = upload_image(image_path)
        print(f"Upload successful!")
        print(f"Image URL: {result}")
        return result

    except Exception as e:
        print(f"Upload failed: {str(e)}")
        return None
