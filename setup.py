from setuptools import find_packages, setup

setup(
    name='trash_detection',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'mlflow>=2.7.0',
        'fastapi>=0.103.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0',
        ],
    },
    author='Rodrigo',
    description='Deep Learning project for trash detection using Computer Vision',
    keywords='computer-vision deep-learning object-detection mlops',
)
