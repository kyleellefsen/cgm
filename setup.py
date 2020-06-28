from setuptools import setup, find_packages
from os import path

pwd = path.abspath(path.dirname(__file__))
with open(path.join(pwd, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# https://github.com/pypa/sampleproject/blob/master/setup.py
setup(
    name='cgm',  
    version='0.0.1',
    description='Causal Graphical Models',
    long_description=long_description, 
    long_description_content_type='text/markdown', 
    url='https://github.com/kyleellefsen/cgm', 
    author='Kyle Ellefsen',
    author_email='kyleellefsen@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='cgm, pgm, dag, causal inference, factors',
    packages=find_packages(where='cgm'), 
    python_requires='>=3.7, <4',
    install_requires=['numpy'],
    project_urls={
        'Source': 'https://github.com/kyleellefsen/cgm',
    },
)