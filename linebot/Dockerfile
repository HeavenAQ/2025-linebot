# Builder
FROM golang:latest AS builder
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
COPY --from=builder /app/pro_videos ./pro_videos

# download the tool needed to run the app
RUN apk update
RUN apk add libc6-compat
RUN apk add --no-cache ffmpeg bash
RUN apk add --no-cache tzdata

# set timezone to Asia/Taipei
ENV TZ="Asia/Taipei"
RUN date

EXPOSE $PORT
CMD ["/root/main"]
