## makefile automates the build and deployment for python projects


## Build
#
PROJ_TYPE =		python
PROJ_MODULES =		git python-resources python-cli python-doc python-doc-deploy
PY_DEP_POST_DEPS +=	modeldeps
ADD_CLEAN_ALL +=	data
ENTRY =			./spanmatch
INT_TEST_LINES =	15


## Includes
#
include ./zenbuild/main.mk


## Targets
#
.PHONY:			modeldeps
modeldeps:
			$(PIP_BIN) install $(PIP_ARGS) -r $(PY_SRC)/requirements-model.txt

.PHONY:			clitest
clitest:
			$(eval cnt = $(shell $(ENTRY) match \
				test-resources/source.txt \
				test-resources/summary.txt \
				-s 2:4 --detail | \
				wc -l))
			@echo "output lines: $(cnt)"
			@if [ $(cnt) -ne $(INT_TEST_LINES) ] ; then \
				echo "expecting $(INT_TEST_LINES) but got $(cnt) output lines" ; \
				false ; \
			fi

.PHONY:			testall
testall:		test clitest
