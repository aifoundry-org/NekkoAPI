# Builder stage
FROM golang:1.24-alpine AS builder

WORKDIR /app
COPY go.* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o nekko-lb ./

FROM scratch
WORKDIR /app
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /app/nekko-lb .
COPY worker-pod-template.yaml /app/templates/

EXPOSE 8080

ENTRYPOINT ["/app/nekko-lb"]
