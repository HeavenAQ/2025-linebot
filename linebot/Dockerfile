# Builder
FROM golang:1.23-alpine AS builder
WORKDIR /app

# download the dependencies first so that they can be cached.
COPY go.mod go.sum ./
RUN go mod download

# copy the source code and build the go app
COPY . .
RUN go build -o main .

# Runner
FROM alpine:latest
WORKDIR /root
COPY --from=builder /app/main .

RUN apk add --no-cache ca-certificates ffmpeg tzdata

# set timezone to Asia/Taipei
ENV TZ="Asia/Taipei"
RUN date

EXPOSE 8080
CMD ["/root/main"]
