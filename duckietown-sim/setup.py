from setuptools import setup


def get_version(filename):
    import ast
    version = None
    with open(filename) as f:
        for line in f:
            if line.startswith('__version__'):
                version = ast.parse(line).body[0].value.s
                break
        else:
            raise ValueError('No version found in %r.' % filename)
    if version is None:
        raise ValueError(filename)
    return version


version = get_version(filename='gym_duckietown/__init__.py')

setup(
        name='gym_duckietown',
        version=version,
        keywords='duckietown, environment, agent, rl, openaigym, openai-gym, gym',
        install_requires=[
            'gym==0.15.4',
            'numpy==1.18.0',
            'pyglet==1.4.10',
            'pyzmq==18.1.1',
            'scikit-image==0.16.2',
            'opencv-python==4.1.2.30',
            'pyyaml==5.2',
            'cloudpickle==1.2.2',
            'duckietown_slimremote==2018.10.1',
            'pygeometry==1.5.6',
            'dataclasses==0.6',
            'torch==1.4.0'
        ],
        entry_points={
            'console_scripts': [
                'duckietown-start-gym=gym_duckietown.launcher:main',
            ],
        },
)
