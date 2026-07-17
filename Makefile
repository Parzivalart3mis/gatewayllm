# GatewayLLM

VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w -X main.version=$(VERSION)

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## Build the gateway and glmctl binaries
	@mkdir -p bin
	go build -ldflags="$(LDFLAGS)" -o bin/gateway ./cmd/gateway
	go build -ldflags="$(LDFLAGS)" -o bin/glmctl ./cmd/glmctl

.PHONY: test
test: ## Run tests with the race detector
	go test -race ./...

.PHONY: test-short
test-short: ## Run tests without the race detector
	go test ./...

.PHONY: cover
cover: ## Report test coverage
	go test -coverprofile=coverage.out ./...
	go tool cover -func=coverage.out | tail -1

.PHONY: lint
lint: ## Vet and check formatting
	go vet ./...
	@test -z "$$(gofmt -l . | tee /dev/stderr)" || (echo "run 'make fmt'"; exit 1)

.PHONY: fmt
fmt: ## Format the code
	gofmt -w .

.PHONY: run
run: ## Run the gateway against the local config
	go run ./cmd/gateway -config config.yaml

.PHONY: up
up: ## Start the full stack
	docker compose up -d --build

.PHONY: down
down: ## Stop the stack
	docker compose down

.PHONY: clean-all
clean-all: ## Stop the stack and delete all data volumes
	docker compose down -v

.PHONY: logs
logs: ## Tail the gateway logs
	docker compose logs -f gateway

.PHONY: seed
seed: ## Create a demo tenant and print a fresh API key
	docker compose exec -T postgres pg_isready -U gateway >/dev/null
	go run ./cmd/glmctl create-tenant -id demo -name "Demo Tenant" \
		-db "postgres://gateway:gateway@localhost:5432/gatewayllm?sslmode=disable"
	go run ./cmd/glmctl create-key -tenant demo -label "demo key" \
		-db "postgres://gateway:gateway@localhost:5432/gatewayllm?sslmode=disable"

.PHONY: clean
clean: ## Remove build artifacts
	rm -rf bin coverage.out
