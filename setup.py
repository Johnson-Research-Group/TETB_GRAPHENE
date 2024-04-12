from setuptools import setup, find_packages

setup(
    name="TETB_GRAPHENE",
    version="0.1",
    author="Daniel Palmer, Naheed Ferdous, Gabriel Brown, Tawfiqur Rakib,Kittithat Krongchon, Lucas K. Wagner, and Harley T. Johnson",
    author_email="dpalmer3@illinois.edu",
    packages=find_packages(),
    install_requires=["joblib","dask","ase","h5py","pandas"],
    include_package_data=True,
)

