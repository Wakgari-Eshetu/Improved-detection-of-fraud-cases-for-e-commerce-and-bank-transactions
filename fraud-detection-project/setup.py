from setuptools import setup, find_packages

setup(
    name='fraud-detection-project',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for improving fraud detection in e-commerce and bank transactions using machine learning techniques.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'imbalanced-learn',
        'matplotlib',
        'seaborn',
        'shap',
        'lime',
        'jupyter',
        'pyyaml',
        'loguru'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)