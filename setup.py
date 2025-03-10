import setuptools
import os
import sys
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(version_path)
from version import __version__
    
setuptools.setup(
    name=__version__,
    version="0.1.0",
    author="N Javed",
    author_email="javed@broadinstitute.org",
    description="predicting cage seq from atac + sequence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/naumanjaved/EpiBERT",
    project_urls={
        "Bug Tracker": "https://github.com/naumanjaved/EpiBERT/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    #package_dir={"": "src"},
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.6",
        install_requires=[
            'tensorflow==2.12.0',
            'numpy==1.23.5',
            'tensorflow-addons==0.23.0',
            'wandb==0.19.8',
            'jax==0.4.30',
            'jaxlib==0.4.30',
            'h5py==3.13.0',
            'scikit-learn==1.6.1',
            'scipy==1.15.1',
            'pandas==2.2.3',
            'matplotlib==3.10.0',
            'seaborn==0.13.2',
            'einops==0.8.1',
            'pysam==0.22.1',
            'pybedtools==0.11.0',
            'fqdn==1.5.1',
            'logomaker==0.8.6',
            'kipoi==0.8.6',
            'tensorboard==2.12.3',
            'tqdm==4.67.1',
            'pycosat==0.6.6',
            'pyfaidx==0.8.1.3',
            'pytabix==0.1',
            'jupyterlab_widgets==3.0.13',
            'bgzip==0.5.0',
        ]
)
