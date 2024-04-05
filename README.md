# 🌱 VigorGraph

## Utility routines

## 🛠 Installation

### 🎈 venv

If you want to create a virtual environment:

`python -m venv path/to/your/dir/.venv`

and source it:

Linux/MacOS:

`source path/to/your/dir/.venv/bin/activate`

Windows:

`path\to\your\dir\.venv\Scripts\Activate`

### 📚 Required libraries

For installing the required libraries (source venv before this):

`pip install -r requirements.txt`

## 🌟 Featuring
- 🌱 Seadling dataset drawer

## 📈 Usage

At [main](./src/main.py), you will have this code

```
from dataset_painter import SeedlingDataset

INPUT_FOLDER = './plantulas_soja/1'
OUTPUT_FOLDER = './dataset/plantulas_soja/1'

if __name__ == '__main__':
    sd = SeedlingDataset(INPUT_FOLDER, OUTPUT_FOLDER)
    sd.run()

```

`INPUT_FOLDER` should be where your images are located, so they would be under `$INPUT_FOLDER/your_image.jpeg`
`OUTPUT_FOLDER` should be where your images will be expored. Primarily they will be exported to `$INPUT_FOLDER/input` and `$INPUT_FOLDER/ground_truth`, such that the `input` would contain the images that you annotated and `ground_truth` would contain the annotations to those images.

Some `$INPUT_FOLDER/input/myimage.jpeg`, would have a correspondent annnotation with this name `$INPUT_FOLDER/ground_truth/myimage.jpeg`.


Key Bindings

- 'r': Set the drawing color to RAIZ PRIMARIA.
- 'h': Set the drawing color to HIPOCOTILO.
- 'd': Activate drawing mode.
- 'e': Activate erasing mode.
- 'xx': Terminate the draw and do not save the current image.
- 'ss': Save the image and goes to the next image.


After that, just run

`python src/main.py`
