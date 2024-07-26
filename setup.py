from setuptools import setup, find_packages

setup(name='ivim',
      version='1.0',
      description='Tools related to Intravoxel Incoherent Motion (IVIM) modeling of diffusion MRI data',
      author='Oscar Jalnefjord',
      author_email='oscar.jalnefjord@gu.se',
      # packages=['ivim','ivim.preproc','ivim.io','ivim.seq'],
      packages=find_packages(),
      install_requires = ['numpy', 'scipy', 'nibabel']
     )
