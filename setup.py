from setuptools import setup, find_packages

setup(
    name='MLFromScratch',
    version='v0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "numpy"
    ],
    package_data={
        "lib": ["*.so"]
    },
    include_package_data=True
)