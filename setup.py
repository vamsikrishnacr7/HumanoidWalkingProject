from setuptools import setup, find_packages

setup(
    name='humanoid_walk_project',  # The name of the package your teammates will import
    version='0.1.0',
    description='Deep Reinforcement Learning for Humanoid Locomotion from Image Pose.',
    author='Your Team Name or Team Lead Name',
    
    # Tells setuptools to find the 'modules' directory and include everything inside.
    packages=find_packages(),
    
    # Listing top-level dependencies. The actual versions are locked in requirements.txt.
    install_requires=[
        'numpy',
        'opencv-python',
        'openpifpaf',
        'pybullet',
        'torch',
        'torchvision',
        'tqdm',
        'matplotlib',
        'gymnasium',
        'scipy',
    ],
)