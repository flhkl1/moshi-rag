set -ex

rm -rf ./moshi/.venv ./moshi/dist ./rust/target
docker compose -f ./swarm-config.yaml build  --push --progress=plain

#docker -H ssh://root@moshi-rag.kyutai.org service update --with-registry-auth  --image rg.fr-par.scw.cloud/namespace-unruffled-tereshkova/moshi-rag-backend:tmp moshi-rag_backend
docker -H ssh://root@moshi-rag.kyutai.org stack deploy -c ./swarm-config.yaml --with-registry-auth --prune moshi-rag
