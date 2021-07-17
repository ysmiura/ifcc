import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='ifcc',
    version='0.2.0',
    author='Yasuhide Miura',
    author_email='ysmiura@stanford.edu',
    description='The code of: Improving Factual Completeness and Consistency of Image-to-text Radiology Report Generation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ysmiura/ifcc',
    packages='clinicgen',
    python_requires='>=3.7',
    install_requires=[
        'bert-score==0.3.0',
        'bioc==1.3.4',
        'bllipparser==2016.9.11',
        'cachetools==4.1.0',
        'flask==1.1.1',
        'jpype1==0.6.3',
        'networkx==1.11',
        'nltk==3.4.5',
        'numpy==1.18.5',
        'pandas==1.0.1',
        'pathlib2==2.3.5',
        'ply==3.11',
        'pystanforddependencies==0.3.1',
        'rouge==0.3.2',
        'scispacy==0.2.0',
        'spacy==2.1.3',
        'stanza==1.1.1',
        'tensorboard==2.0.0',
        'torch==1.5.0',
        'torchvision==0.6.0',
        'tqdm==4.45.0',
        'transformers==2.9.0'
    ]
)
