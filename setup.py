from setuptools import setup, find_packages

setup(
    name="pbe-vae",
    version="1.0.0",
    description="Probability Based Expert Variational Autoencoder Package",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "pbe-vae=pbe_vae.cli.main:main",
        ],
    },
)
