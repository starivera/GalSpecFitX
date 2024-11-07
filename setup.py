from setuptools import setup, find_packages

setup(
    name="GalSpecFitX",
    version="1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'astropy==6.0.0',
        'cvxopt==1.3.2',
        'matplotlib==3.8.2',
        'numpy==1.26.2',
        'git-lfs',
        'ppxf==9.1.1',
        'spectres==2.2.0',
        'lmfit==1.3.1',
        'plotly==5.22.0',
    ],
    entry_points={
        'console_scripts': [
            'galspecfitx=GalSpecFitX.__main__:main',  # Entry point to run GalSpecFitX from CLI
        ],
    },
)
