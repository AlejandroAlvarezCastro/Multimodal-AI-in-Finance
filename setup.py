from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='multimodal_ai_in_finance',
    version='0.1.0',
    packages=find_packages(include=['MULTIMODAL', 'MULTIMODAL.*']),
    install_requires=requirements,
    author='Alejandro Álvarez Castro',
    description='Multimodal AI in Finance',
    license='MIT',
    include_package_data=True,
)
