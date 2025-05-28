# NekkoAPI

OpenAI compatible web API server for llama.cpp.

Disclaimer: this project is in early development stage.
Missing features, bugs and poor documentation are expected.
This should change in the very near future.
Thank you for coming!


## Quick Start

You will need: docker, make, git and curl.

This repository uses submodules, therefore
you should clone it with:
```sh
git clone --recurse-submodules https://github.com/aifoundry-org/NekkoAPI.git
cd NekkoAPI
```

If you have already cloned it and forgot submodules,
you can always initialize them with:
```sh
git submodule update --init --recursive
```

To build NekkoAPI docker image run:

```sh
make docker
```

You can download a few small(ish) (around 5GB total) preconfigured
LLM models from [Hugging Face](https://huggingface.co/) with:

```sh
make example-models
```

### Running the API

With models downloaded you can run the Nekko API server to use them
with:

```sh
make run-example-docker
```

API will be accesible on `http://localhost:8000`.

There is a small example using OpenAPI client library that can
be used to try out the server in `examples/openai-client.py`.

### Running the API with Open WebUI

If you would like to explore the models with an intuitive GUI, run the provided
example script:

```sh
make run-demo
```

Both the API and the UI will be available on `http://localhost`.

> [!WARNING]
> Not every feature of Open WebUI will work at this time. The API server is
> currently unstable and feature-incomplete and may crash if called with
> unsupported parameters.

## Deployment

### With Docker Compose

> [!WARNING]
> At the moment, our images are only built for x86_64 machines. If you need to
> deploy the project to a different architecture, take a look at
> `docker/web/docker-compose.yml` where the image is built from scratch.

* [API-only setup](/examples/docker-compose/api)
* [API + UI + nginx](/examples/docker-compose/ui)


## Development

WIP


## Acknowledgements

This project was made possible thanks to the outstanding work of several
open-source contributors and communities. We would like to express our heartfelt
gratitude to:
- Andrei Betlen, for creating and maintaining llama-cpp-python, which served
  as the foundation for this project. Your efforts in building and maintaining
  this library have been invaluable.
- Georgi Gerganov and the ggml authors, for developing llama.cpp and ggml,
  which are core dependencies of this project. Your innovation and dedication
  to open-source have greatly enriched the ecosystem and enabled
  countless projects, including ours.

Thank you!


## License

This project is licensed under the terms of the Apache License version 2.0.
