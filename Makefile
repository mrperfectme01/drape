.PHONY: install train quick serve test loadtest docker
install:  ; pip install -r requirements.txt
train:    ; python -m scripts.train
quick:    ; python -m scripts.train --quick
serve:    ; uvicorn drape.api.server:app --port 8000
test:     ; python -m pytest -q
loadtest: ; python -m scripts.loadtest --url http://localhost:8000 --n 2000 --concurrency 16
docker:   ; docker build -t drape . && docker run -p 8000:8000 drape
