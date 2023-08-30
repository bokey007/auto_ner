from setuptools import setup, find_packages

# Read the contents of requirements.txt from package root
# with open('requirements.txt') as f:
#     install_requires = f.read().splitlines()

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='auto_ner',
    version='0.1.1',
    author='Sudhir Arvind Deshmukh',
    description='End to End application for named entity recognition. Highlights: 1. Powerd by GenAi 2. Few shot Learning 3. Training and inference pipelines',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/bokey007/auto_ner',
    packages=find_packages(),
    install_requires=[
        'matplotlib==3.6.0',
        'pandas==1.5.3',
        'scikit_learn==1.3.0',
        'spacy==3.6.1',
        'streamlit==1.13.0',
        'transformers==4.32.1',
    ],
    entry_points={
        'console_scripts': [
            'auto_ner.run=auto_ner.run:run',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

#how to build test and bublish this pkg

# pip uninstall imp3
# python setup.py sdist bdist_wheel
# pip install ./dist/imp3-0.1.0.tar.gz
# imp3.run
# twine upload dist/*