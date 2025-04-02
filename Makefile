## MAKEFILE

.PHONY: help
help:
	@echo "	black			: formats code using black, the Python code formatter"
	@echo "	lint			: checks source code with flake8"
	@echo "	mypy			: checks static typing with mypy"
	@echo "	isort			: sorts the imports"
	@echo " tests"          : run unit tests"
	@echo "	docker-build	: build the docker image"

SRC := apple_stock_extremes_ml/

.PHONY: black
black:
	black $(SRC) tests/

.PHONY: lint
lint:
	flake8 --max-line-length 120 --ignore E203,E402,W503  $(SRC)

.PHONY: isort
isort:
	isort  $(SRC)

.PHONY: mypy
mypy:
	mypy --config-file configs/mypy.ini $(SRC)
	rm -rf .mypy_cache

.PHONY: tests
tests:
	# test using the pytest package with the coverage plugin
	# the coverage configuration file is the .coveragerc
	pytest --cov-report term-missing --cov=$(SRC) --cov-config=.coveragerc tests/

	# pytest produces .coverage* files and .pytest_cache
	# pause for 5 seconds until all pytest output files are written to
	# disk (some take a while)
	sleep 5

	# delete all the generated files
	rm -rf .pytest_cache/
	rm -f .coverage
	rm -f .coverage.*


.PHONY: docker-build
docker-build:
	docker build -t apple_stock_extremes_ml .

.PHONY: docker-run


