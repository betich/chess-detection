# chess-detection

## Installation

1. Clone the repository

```bash
git clone https://github.com/betich/chess-detection
```

2. Install the packages

```bash
pip install -r requirements.txt
```

3. Add the video input to the `kaggle/videos` folder

4. Set up Google Cloud Vision API credentials by following the instructions [here](https://cloud.google.com/vision/docs/setup).

5. Download service account key and save it as `service_account.json` in the root directory of the project.

6. Run the script

```bash
python run.py
```

### For the algorithm on Kaggle:

Run `vidhanddetect.py` in `src/chess_board/vidhanddetect.py` which outputs a list of detected chess move which is converted to PGN format from the video.

### For all algorithms (as described in the poster and presentation):

run `run.py` in the root directory of the project.
