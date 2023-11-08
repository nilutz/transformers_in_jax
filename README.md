# Transformers with jax: jaxformers

The idea is start from a vanilla transformer move to the BERT style transformers (the encoding part) and the GPT style transformer(the decoding part). Implement the basic training routines and "replicate" the papers results. Why ? Because basically the underlying modules are all the same in these architectures and its fun to illustrate that fact. All this is just for fun. Keep in mind that a lot has changed and a lot of improvements could have been made, but I try to be as close as possible to the papers.


Currently implemented, tested and run:

* Vanilla - Attention is all you need
* BERT
* GPT

# Setup

If you want to use Jax on Apple M1 devices:
* https://developer.apple.com/metal/jax/

Usual requirements file install

    pip install -r requirements.txt

    pip install -r requirements.dev.txt

For cuda:

    pip install -r requirements.cuda.txt


# Get Started

Logging with tensorboard

    tensorboard --logdir=runs/ --port=6006 --bind_all #http://lernwerk.chickenkiller.com:6006/#


## On Lernwerk

    make sync_files_to_lernwerk

    make build_lernwerk

    make run_gpu0_lernwerk

    source .venv/bin/activate && <a command>

## Format

    black jaxformers

## Docs

Build docs with:

    pdoc --output-dir docs jaxformers --math

# Resources
More resources for each transformer: [here](./transformer_resources.md)