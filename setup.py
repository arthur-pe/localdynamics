from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='localdynamics',
    packages=find_packages(exclude=['tests*']),
    version='0.0.1',
    package_data = {
        '': ['*.ttf'],
    },

    description='Package to perform local dynamics singular value analysis.',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/arthur-pe/localdynamics',
    author='Arthur Pellegrino',
    license='MIT',
    install_requires=['jax',
                      'matplotlib',
                      'finitediffx'
                      ],
    python_requires='>=3.10',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
)