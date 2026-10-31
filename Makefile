PY ?= python
CLI = $(PY) -m eval.cli
CONFIG ?= configs/baseline.yaml
REPEATS ?= 1

.PHONY: install test validate rehash diff ask run score compare gate report history noise offline demo

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest

# --- dataset versioning ----------------------------------------------------
validate:
	$(CLI) dataset validate

rehash:
	$(CLI) dataset rehash

diff:
	$(CLI) dataset diff

# --- agent -----------------------------------------------------------------
ask:
	$(CLI) ask --config $(CONFIG) "$(Q)"

# --- pipeline --------------------------------------------------------------
run:
	$(CLI) run --config $(CONFIG) --repeats $(REPEATS)

score:
	$(CLI) score $(RUN)

compare:
	$(CLI) compare $(A) $(B)

gate:
	$(CLI) gate $(A) $(B)

report:
	$(CLI) report $(A) $(B)

history:
	$(CLI) history

noise:
	$(CLI) judge-noise $(RUN) --times 3 --sample 8

# --- no API key, no cost ---------------------------------------------------
offline:
	$(CLI) run --config configs/offline.yaml
	@echo "Now: make score-offline RUN=<run_id printed above>"

score-offline:
	$(CLI) score $(RUN) --deterministic-only
