from setuptools import setup, find_packages

setup(
    name="GalSpecFitX",
    version="1.0.0",
    author="Isabel Rivera",
    description="Full-spectrum fitting of galaxy spectra using pPXF with automated preprocessing and uncertainty estimation.",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        'astropy==6.0.0',
        'cvxopt==1.3.2',
        'matplotlib==3.8.2',
        'numpy==1.26.2',
        'ppxf>=9.1.1',
        'spectres>=2.2.0',
        'plotly==5.22.0',
        'scipy>=1.14.1',
        "dust-extinction>=1.5.1",
    ],
    entry_points={
        'console_scripts': [
            'galspecfitx=GalSpecFitX.__main__:main',  # Entry point to run GalSpecFitX from CLI
        ],
    },
)
