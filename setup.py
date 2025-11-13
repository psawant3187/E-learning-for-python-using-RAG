import setuptools

setuptools.setup(
    name="text_2_sql_generator_package",
    version="1.0.0",
    author="Pranit Sawant",
    author_email="pranit.sawant.cerelabs@gmail.com",
    description="Text-to-SQL query generator means generating SQL query from natural Language",
    long_description="",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    include_package_data=True,
    install_requires=[
        "openai_helper==2.0.0",
        "Requests==2.32.3",
        "setuptools==72.2.0",
        "queryfic_task_manager_helper>=1.0.9",
        "qdrant_helper>=1.1.0"
    ],
)
