from setuptools import setup, find_packages

setup(
    name='trainer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'joblib'
    ],
    entry_points={
        'console_scripts': []
    },
)
