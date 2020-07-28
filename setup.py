import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="IDSimPy",
    version="0.0.1",
    author="Physical and Theoretical Chemistry, University of Wuppertal",
    author_email="wissdorf@uni-wuppertal.de",
    description="Result analysis and input preprocessing for ion dynamics simulations with IDSimF",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IPAMS/IDSimPy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy','matplotlib','pandas','h5py']
)
