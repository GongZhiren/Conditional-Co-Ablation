PYTHON ?= python
MODEL_KEY ?= gpt2-small
MODEL_PATH ?=
MODEL_PATH_ARG := $(if $(MODEL_PATH),--model-path $(MODEL_PATH),)

.PHONY: install check smoke headline mechanism causal completion knockout panel cross-model

install:
	$(PYTHON) -m pip install -e .

check:
	$(PYTHON) scripts/check_release.py
	$(PYTHON) -m compileall -q src experiments scripts

smoke:
	$(PYTHON) experiments/paper/backup_recovery_full.py --model-key $(MODEL_KEY) \
		$(MODEL_PATH_ARG) --num-prompts 8 --seeds 1 --top-r 192 --skip-grad --suffix _smoke

headline:
	$(PYTHON) experiments/paper/backup_recovery_full.py --model-key $(MODEL_KEY) \
		$(MODEL_PATH_ARG) --num-prompts 96 --seeds 1 15 22 8 --position-mode last --top-r 0

mechanism:
	@for seed in 1 15 22 8; do \
		$(PYTHON) experiments/paper/mechanism_handoff.py --model-key $(MODEL_KEY) \
			$(MODEL_PATH_ARG) --num-prompts 96 --seed $$seed \
			--out outputs/coablation/mechanism_$(MODEL_KEY)_seed$$seed.json || exit 1; \
	done
	$(PYTHON) scripts/summarize_mechanism.py \
		outputs/coablation/mechanism_$(MODEL_KEY)_seed1.json \
		outputs/coablation/mechanism_$(MODEL_KEY)_seed15.json \
		outputs/coablation/mechanism_$(MODEL_KEY)_seed22.json \
		outputs/coablation/mechanism_$(MODEL_KEY)_seed8.json \
		--out outputs/coablation/mechanism_$(MODEL_KEY)_summary.json

causal:
	$(PYTHON) experiments/paper/causal_freezing.py --model-key $(MODEL_KEY) \
		$(MODEL_PATH_ARG) --num-prompts 96 --seeds 1 15 22 8 --k 8

completion:
	$(PYTHON) experiments/paper/circuit_completion.py --model-key $(MODEL_KEY) \
		$(MODEL_PATH_ARG) --num-prompts 96 --seeds 1 15 22 8 --top-r 0

knockout:
	$(PYTHON) experiments/paper/knockout_oracle_distance.py --model-key $(MODEL_KEY) \
		$(MODEL_PATH_ARG) --num-prompts 96 --seeds 1 15 22 8 --topk 8

panel:
	$(PYTHON) experiments/paper/matched_intervention_panel.py --model-key $(MODEL_KEY) \
		$(MODEL_PATH_ARG) --num-prompts 96 --ind-seqs 32 --seq-len 40 \
		--calib-seed 1 --valid-seed 15 --position-mode last --top-r 0 \
		--bootstrap 20000 --resume

cross-model:
	$(PYTHON) experiments/paper/cross_model_completion.py --model-key $(MODEL_KEY) \
		$(MODEL_PATH_ARG) --n-detect 32 --n-calib 16 --n-eval 64 --topk 10
