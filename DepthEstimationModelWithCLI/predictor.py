import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)


def colorize(depth_map):
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    colored = plt.cm.plasma(depth_norm)
    colored = (colored[:, :, :3] * 255).astype(np.uint8)
    return colored


class DepthEstimationModel:
    def __init__(self):
        self.model = self._initialize_model().to(self._get_device())

    def _get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _initialize_model(self):
        try:
            torch.hub._get_cache_dir()

            model = torch.hub.load(
                "isl-org/ZoeDepth",
                "ZoeD_N",
                pretrained=True,
                skip_validation=False,
                force_reload=False,
            )

            self._fix_model_state_dict(model)

            model.eval()
            print("Model initialized successfully.")
            return model

        except Exception as e:
            print(f"ZoeD_N failed, trying alternative approach: {e}")
            return self._initialize_alternative_model()

    def _fix_model_state_dict(self, model):
        try:
            state_dict = model.state_dict()
            keys_to_remove = [
                k for k in state_dict.keys() if "relative_position_index" in k
            ]

            for key in keys_to_remove:
                del state_dict[key]

            model.load_state_dict(state_dict, strict=False)
            print(f"Removed {len(keys_to_remove)} incompatible parameters")

        except Exception as e:
            print(f"Could not fix state_dict: {e}")

    def _initialize_alternative_model(self):
        try:
            print("Trying MiDaS model as fallback...")
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=True)
            model.eval()
            print("MiDaS model initialized successfully.")
            return model

        except Exception as e:
            print(f"All model loading attempts failed: {e}")
            raise RuntimeError("Could not initialize any depth estimation model")

    def save_colored_depth(self, depth_numpy, output_path):
        try:
            colored = colorize(depth_numpy)
            Image.fromarray(colored).save(output_path)
            print(f"Colored depth image saved to {output_path}")
        except Exception as e:
            print(f"Error saving colored depth: {e}")

    def calculate_depthmap(self, image_path, output_path):
        try:
            image = Image.open(image_path).convert("RGB")
            print("Image loaded successfully.")

            if hasattr(self.model, "infer_pil"):
                depth_numpy = self.model.infer_pil(image)
            else:
                depth_numpy = self._infer_midas(image)

            self.save_colored_depth(depth_numpy, output_path)

            raw_output_path = output_path.replace(".png", "_raw.npy")
            np.save(raw_output_path, depth_numpy)
            print(f"Raw depth data saved to {raw_output_path}")

            return f"Depth map saved to {output_path}"

        except Exception as e:
            print(f"Error in depth calculation: {e}")
            return None

    def _infer_midas(self, image):
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        input_tensor = transform(image).unsqueeze(0).to(self._get_device())

        with torch.no_grad():
            depth = self.model(input_tensor)
            depth = depth.squeeze().cpu().numpy()

        return depth


if __name__ == "__main__":
    try:
        print("Initializing Depth Estimation Model...")
        model = DepthEstimationModel()

        import os

        if not os.path.exists("./test.png"):
            print(
                "Warning: test.png not found. Please make sure the image file exists."
            )
        else:
            print("Processing depth map...")
            result = model.calculate_depthmap("./test.png", "output_depth_map.png")
            if result:
                print(f"Success: {result}")
            else:
                print("Failed to process depth map")

    except Exception as e:
        print(f"Critical error: {e}")

        print("\nDebug Information:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Cache directory: {torch.hub.get_dir()}")
