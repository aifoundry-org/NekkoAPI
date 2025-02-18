# Quick start

This folder contains an example kubernetes config (manifests) to deploy
NekkoAPI together with prometheus and graphana.

First you need to build the docker image with example LLM models included.
This can be accoplished by running:

```sh
make requirements
```

This downloads example models (Llama-3.2-1B and Smollm2-135m), builds
NekkoAPI runtime and produces a docker image `nekko-api-models`.

You need to have k8s cluster up and running with `kubectl` configured.

Depending on your k8s environment you may need to upload the image
to the registry. Once image is available on your k8s cluster, you
can deploy everything with:

```sh
kubectl apply -f manifests/
```

(Services are exposed using `LoadBalancer` type of Service resource -
please edit accordingly if it is not available on your system.)

Once deployed, Nekko API should be available on port `3080`. You can
find external IP by running:

```sh
kubectl get svc
```

Disclaimer: This is just an example configuration for testing and devlopment
of NekkoAPI and is not intented for production use.
