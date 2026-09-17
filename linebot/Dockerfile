# Builder
FROM golang:1.23-alpine AS builder
WORKDIR /app

# download the dependencies first so that they can be cached.
COPY go.mod go.sum ./
RUN go mod download

# copy the source code and build the go app
COPY . .
RUN CGO_ENABLED=0 go build -o main .

# Runner
FROM alpine:latest
RUN apk add --no-cache ca-certificates ffmpeg tzdata && \
    addgroup -S -g 10001 app && \
    adduser -S -D -H -u 10001 -G app -h /app app && \
    mkdir -p /app && chown app:app /app

# set timezone to Asia/Taipei
ENV TZ="Asia/Taipei"

# Run unprivileged. The working directory must stay writable by that user:
# on startup the server writes the .env it fetches from Secret Manager there,
# and thumbnails are extracted under /tmp.
WORKDIR /app
COPY --from=builder --chown=app:app /app/main /app/main
USER app:app

EXPOSE 8080
CMD ["/app/main"]
