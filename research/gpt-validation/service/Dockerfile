# syntax=docker/dockerfile:1

FROM golang:1.23-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/gpt-validation .

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/gpt-validation /gpt-validation
USER nonroot:nonroot
ENV PORT=8080
EXPOSE 8080
ENTRYPOINT ["/gpt-validation"]
