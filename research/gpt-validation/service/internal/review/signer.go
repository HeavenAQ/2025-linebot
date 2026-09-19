package review

import (
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"

	"cloud.google.com/go/storage"
)

// SignedURLTTL is how long a video link stays valid.
const SignedURLTTL = 60 * time.Minute

// MediaPrefix is the only object tree this service may sign.
const MediaPrefix = "gpt-validation/"

// Signer mints read URLs for stored objects.
type Signer interface {
	SignedURL(objectPath string) (string, error)
}

// AllowedObjectPath reports whether objectPath may be signed: it must live
// under gpt-validation/ and must not contain "..".
func AllowedObjectPath(objectPath string) bool {
	return strings.HasPrefix(objectPath, MediaPrefix) &&
		!strings.Contains(objectPath, "..") &&
		!strings.ContainsAny(objectPath, "\x00\r\n")
}

// normalizeObjectPath accepts a bare object path or a gs:// URI in bucket.
func normalizeObjectPath(bucket, path string) (string, bool) {
	path = strings.TrimSpace(path)
	if rest, ok := strings.CutPrefix(path, "gs://"); ok {
		b, obj, found := strings.Cut(rest, "/")
		if !found || b != bucket {
			return "", false
		}
		path = obj
	}
	return path, AllowedObjectPath(path)
}

// GCSSigner signs V4 GET URLs. With no key file (Cloud Run) the storage client
// signs through IAM SignBlob as GoogleAccessID, which needs
// roles/iam.serviceAccountTokenCreator on that service account.
type GCSSigner struct {
	client              *storage.Client
	bucket              string
	serviceAccountEmail string
	now                 func() time.Time
}

// NewGCSSigner builds a signer for one bucket.
func NewGCSSigner(client *storage.Client, bucket, serviceAccountEmail string) *GCSSigner {
	return &GCSSigner{client: client, bucket: bucket, serviceAccountEmail: serviceAccountEmail, now: time.Now}
}

// SignedURL signs objectPath after checking the allowlist.
func (g *GCSSigner) SignedURL(objectPath string) (string, error) {
	path, ok := normalizeObjectPath(g.bucket, objectPath)
	if !ok {
		return "", fmt.Errorf("object path is not signable: %q", objectPath)
	}
	opts := &storage.SignedURLOptions{
		Method:  "GET",
		Expires: g.now().Add(SignedURLTTL),
		Scheme:  storage.SigningSchemeV4,
		// Objects may lack a Content-Type; force a playable one.
		QueryParameters: url.Values{"response-content-type": []string{"video/mp4"}},
	}
	if email := strings.TrimSpace(g.serviceAccountEmail); email != "" {
		opts.GoogleAccessID = email
	}
	signed, err := g.client.Bucket(g.bucket).SignedURL(path, opts)
	if err != nil {
		return "", fmt.Errorf("sign %q: %w", path, err)
	}
	return signed, nil
}

// NoopSigner is used with fake local data: it never produces URLs.
type NoopSigner struct{}

// SignedURL always fails (after enforcing the same allowlist).
func (NoopSigner) SignedURL(objectPath string) (string, error) {
	if !AllowedObjectPath(objectPath) {
		return "", fmt.Errorf("object path is not signable: %q", objectPath)
	}
	return "", errors.New("signing disabled in local fake mode")
}
