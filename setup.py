from setuptools import setup, find_packages

setup(
    name="eaf_base_api",
    version="2.2.3",
    packages=find_packages(),
    install_requires=["ffmpeg-progress-yield", "m3u8", "httpx"],
    entry_points={
        'console_scripts': [
            # If you want to create any executable scripts
        ],
    },
    author="Johannes Habel",
    author_email="EchterAlsFake@proton.me",
    description="A base API for EchterAlsFake's Porn APIs",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license="LGPLv3",
    url="https://github.com/EchterAlsFake/eaf_base_api",
    classifiers=[
        # Classifiers help users find your project on PyPI
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Programming Language :: Python",
    ],
)