from setuptools import setup, find_packages

setup(
    name="asl_cam",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.64.0",
    ],
    python_requires=">=3.8",
    author="Raul Adell",
    description="Classical computer vision approach to ASL hand detection",
    long_description="A build-it-yourself ASL hand detection system using classical CV techniques",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
) 