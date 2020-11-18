# open-tutor-classifier

A short answer classifier service focused on cold-start performance


## Requirements

- python3.8 (must be in path as `python3.8` to build virtualenv)
- make

## Development

Any changes made to this repo should be covered by tests. To run the existing tests:

```
# NOTE: the first time you run this will take a few minutes
# because it downloads the 300+MB word2vec file one time
make test
```

All pushed commits must also pass format, lint, and license checks. To check all required tests before a commit:

```
make test-all
```

To fix formatting issues:

```
make format
```

## Licensing

All source code files must include a USC open license header.

To check if files have a license header:

```
make test-license
```

To add license headers:

```
make license
```
