# open-tutor-classifier

A short answer classifier service focused on cold-start performance


## Requirements

- python3.8 (must be in path as `python3.8` to build virtualenv)
- make

## Development

Any changes made to this repo should be covered by tests. To run the existing tests:

```
make test
```

All pushed commits must also pass format and lint checks. To check all required tests before a commit:

```
make test-all
```

To fix formatting issues:

```
make format
```