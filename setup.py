from setuptools import setup

setup(
    name='property-estimator',
    install_requires=[
        'protobuf==3.20.3',
        'grpcio==1.51.1',
        'firebase-admin>=6.0.0,<7.0.0',
        'flask>=2.0.0,<3.0.0'
    ],
    python_requires='>=3.9',
)