from os import major
from setuptools import setup
from setuptools import find_packages
import os.path


name = "batchflow_easyocr"
install_requires = []

setup(
    name=name,
    namespace_packages=["batchflow_hub"],
    version="0.1",
    description="Yolo v7",
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
