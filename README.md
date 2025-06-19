# ASL Camera Recognition

Real-time American Sign Language recognition using computer vision and deep learning.

## Features (Planned)

- Real-time webcam hand tracking
- ASL gesture recognition
- Translation to English text
- Support for basic ASL signs
- Interactive learning mode

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RaulAdSe/cv-asl.git
cd cv-asl
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Development Setup

1. Install development dependencies:
```bash
pip install pytest pytest-mock mypy ruff
```

2. Run tests:
```bash
pytest
```

3. Run type checking:
```bash
mypy src/asl_cam
```

4. Run linting:
```bash
ruff check src/
```

## Project Structure

```
src/
├── asl_cam/
│   ├── config/              # Configuration files
│   ├── data/                # Dataset samples
│   ├── utils/               # Helper functions
│   ├── capture.py           # Webcam handling
│   ├── preprocess.py        # Image preprocessing
│   ├── dataset.py           # Data loading
│   ├── model.py             # Neural network model
│   ├── train.py             # Training script
│   ├── infer.py            # Real-time inference
│   └── translate.py         # ASL to text translation
└── tests/                   # Unit tests
```

## Usage

To start the webcam capture and recognition:

```bash
python -m asl_cam.capture
```

## Next Steps

- [ ] Implement basic webcam capture
- [ ] Add hand detection using OpenCV
- [ ] Create initial CNN model
- [ ] Collect training data
- [ ] Implement real-time inference

## License

MIT License
