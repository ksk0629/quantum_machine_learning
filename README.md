# INTRODUCTION
This is an exmple project for packaging my own python project.

# PRELIMINARIES
- Install `build` if you need to build the project: `pip install build`.

# HOW TO INSTALL

## FOR USERS
Run the following command to install the package.

`poetry install --only-root`

## FOR DISTRUBUTERS
Run the following command to build the package.

`poetry build`

Then `dist` directory will be created under the root in which the package is installed by the following command.

`poetry add dist/template_py_package-0.1.0-py3-none-any.whl`

The filename depends on the settings written in the `pyproject.toml`.

## FOR DEVELOPERS
Run the following command to install the package and the dependencies.

`poetry install`

# HOW TO RUN TESTS
Run the following command.

`poetry run pytest`.

The following command would be preferred if you need the coverage-report.

`poetry run pytest --cov=. --cov-report term-missing`

for seeing the report on your console, or

`poetry run pytest --cov=. --cov-report=html`

for outputting html files.


# REFERENCE
- [Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [pytest：フィクスチャ(fixture)の使い方](https://qiita.com/_akiyama_/items/9ead227227d669b0564e) (Japanese)
- [GitHub Actions integration](https://black.readthedocs.io/en/stable/integrations/github_actions.html)
- [How to use fixtures](https://docs.pytest.org/en/7.1.x/how-to/fixtures.html)
