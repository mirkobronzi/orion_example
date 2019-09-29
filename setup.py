from setuptools import setup, find_packages


setup(
    name='orion_example',
    version='0.0.1',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    python_requires='>=3.6, <=3.7',

    install_requires=[
        'mlflow', 'orion', 'pyyaml'
    ]
)
