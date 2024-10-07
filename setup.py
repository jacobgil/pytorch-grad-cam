import setuptools

with open('README.md', mode='r', encoding='utf-8') as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

setuptools.setup(
    name='grad-cam',
    version='1.5.4',
    author='Jacob Gildenblat',
    author_email='jacob.gildenblat@gmail.com',
    description='Many Class Activation Map methods implemented in Pytorch for classification, segmentation, object detection and more',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jacobgil/pytorch-grad-cam',
    project_urls={
        'Bug Tracker': 'https://github.com/jacobgil/pytorch-grad-cam/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    packages=setuptools.find_packages(
        exclude=["*tutorials*"]),
    python_requires='>=3.8',
    install_requires=requirements)
