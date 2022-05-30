from distutils.core import setup

setup(
    name='dp_policy',
    version='1.0',
    packages=['dp_policy'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    install_requires=[
        'pandas',
        'matplotlib>=3.4.1',
        'numpy',
        'diffprivlib',
        'geopandas',
        'tqdm',
        'click',
        'seaborn',
        'pyarrow',
        'xlrd',
        'openpyxl'
    ],
    entry_points={
        'console_scripts': ['dp_policy = dp_policy.api:cli']
    }
)
