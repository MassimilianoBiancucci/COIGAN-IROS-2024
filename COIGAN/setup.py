#!/usr/bin/env python

from setuptools import setup, find_packages

requirements = [
    "numpy",
    "opencv-python-headless",
]

test_requirements = []

setup(
    author="Massimiliano Biancucci",
    author_email="Binacucci95@Gmail.com",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
    ],
    description="Training, inference and test pipeline package for a controled masked adversarial generator",
    install_requires=requirements,
    long_description="",
    include_package_data=True,
    keywords="defect_unet",
    name="defect_unet",
    packages=find_packages(
        include=[
            "configs",
            "configs.*",
            "losses",
            "losses.*",
            "models",
            "models.*",
            "utils",
            "utils.*",
        ]
    ),
    dependency_links=[],
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/cloe-ai/Defect-Unet-api",
    version="0.1.0",
    zip_safe=False,
)