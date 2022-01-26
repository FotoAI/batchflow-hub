from os import major
from setuptools import setup
from setuptools import find_packages

name = "batchflow_opencv_facedetector"
install_requires = [
    "numpy>=1.14.0",
    "Pillow>=5.2.0",
    "opencv-python>=3.4.4",
]

setup(
    name=name,
    namespace_packages=["batchflow_hub"],
    version="0.1",
    description="OpenCV face detector",
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
