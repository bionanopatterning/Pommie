from setuptools import setup, find_packages

# push to pypi:
# python setup.py sdist
# twine upload dist/*

setup(
    name='Pommie',
    version='0.0.1',
    packages=find_packages(),
    license='GPL v3',
    author='mgflast',
    author_email='m.g.f.last@lumc.nl',
    long_description_content_type="text/markdown",
    package_data={'': ['*.png', '*.glsl', '*.pdf', '*.txt', '*.json']},
    include_package_data=False,  # weirdly, the above filetypes _are_ included when this parameter is set to False.
    install_requires=[
        "numpy-quaternion>=2022.4.4",
        "glfw",
        "matplotlib",
        "scipy",
        "mrcfile",
        "PyOpenGL"
    ]
)
