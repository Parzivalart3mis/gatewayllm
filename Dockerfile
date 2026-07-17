# --- build ---
FROM golang:1.22-alpine AS build

WORKDIR /src

# Dependencies first so the module cache survives source edits.
COPY go.mod go.sum ./
RUN go mod download

COPY . .

ARG VERSION=dev
# CGO_ENABLED=0 produces a static binary, which is what allows the final stage
# to be a scratch image with no libc at all.
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w -X main.version=${VERSION}" \
    -o /out/gateway ./cmd/gateway

# --- runtime ---
FROM alpine:3.21

# ca-certificates is required: every provider is an HTTPS endpoint, and without
# the trust store TLS verification fails on the first upstream call.
RUN apk add --no-cache ca-certificates tzdata && \
    adduser -D -u 1000 gateway

COPY --from=build /out/gateway /usr/local/bin/gateway

USER gateway
WORKDIR /home/gateway

EXPOSE 8080 9090

ENTRYPOINT ["/usr/local/bin/gateway"]
CMD ["-config", "/etc/gatewayllm/config.yaml"]
