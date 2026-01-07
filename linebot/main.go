package main

import (
    "log"
    "net/http"
    "time"

    "github.com/HeavenAQ/nstc-linebot-2025/app"
    "github.com/gin-gonic/gin"
)

func main() {
    application := app.NewApp(".env")

    r := gin.New()
    r.Use(gin.Recovery())

    // Routes (parity with previous net/http handlers)
    r.POST("/callback", func(c *gin.Context) {
        handler := application.LineWebhookHandler()
        handler(c.Writer, c.Request)
    })
    r.GET("/test", func(c *gin.Context) { c.String(http.StatusOK, "Hello, World!") })

    // Swagger docs removed per user request.

    // HTTP server with timeouts
    const (
        DefaultReadTimeout  = 100 * time.Second
        DefaultWriteTimeout = 100 * time.Second
        DefaultIdleTimeout  = 120 * time.Second
    )
    srv := &http.Server{
        Addr:         "0.0.0.0:" + application.Config.Port,
        Handler:      r,
        ReadTimeout:  DefaultReadTimeout,
        WriteTimeout: DefaultWriteTimeout,
        IdleTimeout:  DefaultIdleTimeout,
    }

    application.Logger.Info.Println("\n\tServer started on port " + application.Config.Port)
    if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
        log.Fatal(err)
    }
}
