from setuptools import setup, find_packages

setup(
    name='tcrdistgpu',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'pandas','numpy','pytest'
    ],
    include_package_data=True,
     package_data={
        'tcrdistgpu': ['data/*.tsv','data/*.csv'],
    },
)