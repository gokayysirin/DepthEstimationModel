import argparse

from predictor import DepthEstimationModel


def main():
    parser = argparse.ArgumentParser(description="Depth Estimation CLI")
    parser.add_argument(
        "image_path", type=str, help="Path to the input image for depth estimation"
    )
    parser.add_argument(
        "output_path", type=str, help="Path to save the output depth map image"
    )
    args = parser.parse_args()
    model = DepthEstimationModel()
    result = model.calculate_depthmap(args.image_path, args.output_path)
    print(result)


if __name__ == "__main__":
    main()
# This code is a command-line interface (CLI) for depth estimation using a pre-trained model.
