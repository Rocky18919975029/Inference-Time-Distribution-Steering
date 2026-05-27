from setuptools import setup, find_packages
import os

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

# with open(os.path.join(version_folder, 'verl/version/version')) as f:
#     __version__ = f.read().strip()


with open('requirements.txt') as f:
    required = f.read().splitlines()
    install_requires = [item.strip() for item in required if item.strip()[0] != '#']

extras_require = {
    'test': ['pytest', 'yapf']
}

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='verl',
    package_dir={'': '.'},
    packages=find_packages(where='.'),
    url='https://github.com/volcengine/verl',
    license='Apache 2.0',
    install_requires=install_requires,
    extras_require=extras_require,
    long_description=long_description,
    long_description_content_type='text/markdown'
)