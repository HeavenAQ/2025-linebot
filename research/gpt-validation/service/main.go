// Command gpt-validation serves the expert review web app for validating GPT
// coaching feedback (see ../DESIGN.md).
package main

import (
	"context"
	"embed"
	"errors"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"cloud.google.com/go/firestore"
	"cloud.google.com/go/storage"

	"github.com/HeavenAQ/nstc-linebot-2025/research/gpt-validation/service/internal/review"
)

//go:embed web
var webFiles embed.FS

func main() {
	logger := review.NewLogger(os.Stdout)
	if err := run(logger); err != nil {
		logger.Error("server stopped", slog.String("error", err.Error()))
		os.Exit(1)
	}
}

func run(logger *slog.Logger) error {
	cfg, err := review.LoadConfig(os.Getenv)
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	var (
		store  review.Store
		signer review.Signer
	)
	if cfg.FakeData {
		logger.Warn("LOCAL_FAKE_DATA=1: serving in-memory sample data; never use in production")
		store = review.NewMemoryStore(review.FakeItems(cfg.BatchID)...)
		signer = review.NoopSigner{}
	} else {
		fsClient, err := firestore.NewClient(ctx, cfg.ProjectID)
		if err != nil {
			return err
		}
		defer fsClient.Close()
		gcsClient, err := storage.NewClient(ctx)
		if err != nil {
			return err
		}
		defer gcsClient.Close()
		store = review.NewFirestoreStore(fsClient)
		signer = review.NewGCSSigner(gcsClient, cfg.BucketName, cfg.ServiceAccountEmail)
	}

	static, err := fs.Sub(webFiles, "web")
	if err != nil {
		return err
	}
	srv := review.NewServer(cfg.BatchID, store, signer, review.NewAuthenticator(cfg.AccessCodes), static, logger)

	httpServer := &http.Server{
		Addr:              ":" + cfg.Port,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      60 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		logger.Info("listening", slog.String("port", cfg.Port), slog.String("batch_id", cfg.BatchID))
		errCh <- httpServer.ListenAndServe()
	}()

	select {
	case err := <-errCh:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
	}

	logger.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return httpServer.Shutdown(shutdownCtx)
}
