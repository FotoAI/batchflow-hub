from os import major
from setuptools import setup
from setuptools import find_packages
import os.path

try:
    import onnxruntime
except:
    raise Exception("Install onnxruntime or onnxruntime-gpu as per the system")

name = "batchflow_insightface"
install_requires = [
    "numpy>=1.14.0",
    "tqdm>=4.30.0",
    "gdown>=3.10.1",
    "Pillow>=5.2.0",
    "opencv-python>=3.4.4",
    "Cython>=0.29.28",
    "insightface==0.6.2",
]

setup(
    name=name,
    namespace_packages=["batchflow_hub"],
    version="0.2",
    description="InsightFace FaceAnalysis",
    author="Tushar Kolhe",
    author_email="tusharkolhe08@gmail.com",
    url="",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        "visualize": ["pydot>=1.2.0"],
        "tests": ["pytest", "pytest-pep8", "pytest-xdist", "pytest-cov"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
