from setuptools import setup, find_packages

setup(
    name="brain_observatory_utilities",
    version="0.1.9",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    description="Utilities for analyzing, manipulating and visualizing data for Brain Observatory projects",
    url="https://github.com/AllenInstitute/mindscope_utilities",
    author="Allen Institute",
    author_email="corbettb@alleninstitute.org, michaelbu@alleninstitute.org, marinag@alleninstitute.org, clark.roll@alleninstitute.org",
    license="Allen Institute",
    install_requires=[
        "flake8",
        "pytest",
        "allensdk",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: Other/Proprietary License",  # Allen Institute Software License
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
    ],
)
