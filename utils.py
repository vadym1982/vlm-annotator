import os


def get_images(path_to_data):
    """
    Scans a directory and returns a list of full paths to supported image files:
    *.jpg, *.jpeg, *.png, *.webp.

    Args:
        path_to_data (str): Path to the directory containing image files.

    Returns:
        list[str]: A list of full paths to the discovered images. If no valid
            images are found, returns an empty list.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')

    images = [
        os.path.join(path_to_data, f)
        for f in os.listdir(path_to_data)
        if f.lower().endswith(valid_extensions)
    ]

    return images
