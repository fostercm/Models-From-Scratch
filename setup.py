from setuptools import setup, find_packages

setup(
    name='MLFromScratch',
    version='v0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "numpy"
    ],
    include_package_data=True,
    package_data={
        'Shared Libraries': ["build/lib/*.so"]
    }
)