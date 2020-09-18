import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DeepGrowth", # Replace with your own username
    version="0.0.1",
    author="Douwe Spaanderman",
    author_email="dspaande@broadinstitute.org",
    description="Machine Learning approach for growing cell lines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Douwe-Spaanderman/Broad_DJ_AI",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)