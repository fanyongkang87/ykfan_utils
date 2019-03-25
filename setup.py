import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ykfan_utils",
    version="0.0.5",
    author="ykfan",
    author_email="fanyongkang87@qq.com",
    description="ykfan utils for deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fanyongkang87/ykfan_utils.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)