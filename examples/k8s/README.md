# Quick start

This folder contains an example kubernetes config (manifests) to deploy
NekkoAPI together with prometheus and graphana. It depends on
Ainekko platform docker images being available on docker hub.
Specifically:
- `aifoundryorg/storage-manager`
- `aifoundryorg/nekko-api`
- `aifoundryorg/load-balancer`

TODO: scripts demonstrating how to control the cluster (instantiate
workers with models from eg. Hugging Face)


## Requirements

You need to have k8s cluster up and running with `kubectl` configured.

TODO: example how to setup minikube cluster


## Deploy

To deploy basic ainekko system on your k8s cluster you just run:

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

TODO: port forwarding instruction

Disclaimer: This is just an example configuration for testing and development of
NekkoAPI and is not intented for production use.
