from setuptools import setup, find_packages

setup(
    name="nco_lib",
    version="0.1.0",
    author="Andoni Irazusta Garmendia",
    author_email="andonirazusta@gmail.com",
    description="A Neural Combinatorial Optimization library based on PyTorch",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TheLeprechaun25/NCOLib",
    package_dir={"": "nco_lib"},
    packages=find_packages(where="nco_lib"),
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
