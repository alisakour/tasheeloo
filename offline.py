from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
from PIL import Image

if __name__ == "__main__":
    fe = FeatureExtractor()

    # Use Path to create the output directory if it doesn't exist
    output_dir = Path("./static/feature")
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(Path("./static/img").glob("*.png")):
        print(img_path)
        try:
            feature = fe.extract(img=Image.open(img_path))
            feature_path = output_dir / f"{img_path.stem}.npy"
            print(feature_path)
            np.save(feature_path, feature)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
