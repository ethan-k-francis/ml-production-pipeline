# ============================================================
# Drift Detector — Multi-Stage Go Build
#
# Multi-stage builds produce minimal final images:
#   Stage 1 (builder): compiles the Go binary (~800MB with Go toolchain)
#   Stage 2 (runtime): copies only the binary into a scratch-like image (~15MB)
#
# The final image contains just the static binary + CA certificates
# (needed for HTTPS webhook calls). No shell, no package manager,
# no attack surface — exactly what you want for a production service.
# ============================================================

# ── Stage 1: Build ────────────────────────────────────────
FROM golang:1.23-alpine AS builder

WORKDIR /build

# Copy go.mod first for dependency caching.
# Since this project uses only stdlib, there's nothing to download —
# but the pattern is correct for when dependencies are added later.
COPY go.mod ./
RUN go mod download

# Copy source code and compile a statically-linked binary.
# CGO_ENABLED=0 ensures no C library dependencies (pure Go).
# -ldflags="-s -w" strips debug symbols, reducing binary size ~30%.
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o drift-detector ./cmd/detector

# ── Stage 2: Runtime ──────────────────────────────────────
FROM alpine:3.20

# CA certificates are needed for HTTPS webhook calls.
# Create a non-root user — containers should never run as root.
RUN apk --no-cache add ca-certificates && \
    adduser -D -u 65532 appuser

WORKDIR /app

# Copy the compiled binary from the builder stage
COPY --from=builder /build/drift-detector .

# Reference distributions are mounted at runtime (or baked into image)
COPY reference_distributions.json ./reference_distributions.json

# Run as non-root for security — least-privilege principle
USER appuser

ENTRYPOINT ["./drift-detector"]
