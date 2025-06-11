from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="multimodal_fin",                # <— change this line
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "multimodal-fin = multimodal_fin.cli:cli",  # <— and this
        ],
    },
)
