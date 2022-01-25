from os import major
from setuptools import setup
from setuptools import find_packages
import os.path

try:
    import tensorflow as tf
    version = tf.__version__
    version, major_version, minor_version = version.split(".")
    if int(version) <= 1 and (int(major) < 9):
        raise Exception("Install tensorflow >1.9.0 cpu or gpu as per your system")
except:
    raise Exception("Install tensorflow >1.9.0 cpu or gpu as per your system")

name = "batchflow_arcface"
install_requires = [
    "numpy>=1.14.0",
    "pandas>=0.23.4",
    "tqdm>=4.30.0",
    "gdown>=3.10.1",
    "Pillow>=5.2.0",
    "opencv-python>=3.4.4",
    "keras>=2.2.0"
]

setup(
    name=name,
    namespace_packages=["batchflow_hub"],
    version="0.1",
    description="ArcFace Face Encoding",
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
