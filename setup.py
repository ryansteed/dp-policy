from distutils.core import setup

setup(
    name='dp_policy',
    version='0.1',
    packages=['dp_policy',],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    install_requires=[
        'pandas',
        'matplotlib',
        'numpy',
        'diffprivlib'
    ]
)
