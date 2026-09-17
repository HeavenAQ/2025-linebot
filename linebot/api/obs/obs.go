// Package obs writes structured logs Cloud Logging understands and carries a
// request's correlation IDs (request ID and Cloud Trace context) through
// context.Context, so the Go backend's logs, its calls to the GPU analysis
// service, and that service's logs all share one trace.
package obs

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// TraceHeader is the header Cloud Run's front end sets on every request, in
// the form TRACE_ID/SPAN_ID;o=OPTIONS. Forwarding it lets Cloud Trace join the
// analysis service's request span to the caller's.
const TraceHeader = "X-Cloud-Trace-Context"

// RequestIDHeader lets a caller supply its own correlation ID.
const RequestIDHeader = "X-Request-Id"

type contextKey int

const requestKey contextKey = iota

// Request holds the correlation IDs of one unit of work.
type Request struct {
	ID      string
	TraceID string
	SpanID  string
	Sampled bool
}

var traceHeaderPattern = regexp.MustCompile(`^([0-9a-fA-F]{32})(?:/([0-9]{1,20}))?(?:;o=([01]))?$`)

// ParseTraceHeader reads an X-Cloud-Trace-Context value; malformed input
// yields ok=false rather than a partial trace.
func ParseTraceHeader(value string) (traceID, spanID string, sampled, ok bool) {
	match := traceHeaderPattern.FindStringSubmatch(strings.TrimSpace(value))
	if match == nil {
		return "", "", false, false
	}
	return strings.ToLower(match[1]), match[2], match[3] == "1", true
}

// NewID returns a random 32-hex-character ID, which is also a valid trace ID.
func NewID() string {
	var buffer [16]byte
	if _, err := rand.Read(buffer[:]); err != nil {
		return strconv.FormatInt(time.Now().UnixNano(), 16)
	}
	return hex.EncodeToString(buffer[:])
}

// FromHeaders builds the request's correlation IDs from inbound headers,
// starting a new trace when the front end supplied none.
func FromHeaders(traceHeader, requestID string) Request {
	request := Request{ID: sanitizeID(requestID)}
	if traceID, spanID, sampled, ok := ParseTraceHeader(traceHeader); ok {
		request.TraceID, request.SpanID, request.Sampled = traceID, spanID, sampled
	} else {
		request.TraceID = NewID()
	}
	if request.ID == "" {
		request.ID = request.TraceID
	}
	return request
}

// A client-supplied request ID ends up in logs and gRPC metadata, so keep it
// short and printable.
func sanitizeID(value string) string {
	value = strings.TrimSpace(value)
	if len(value) > 128 {
		value = value[:128]
	}
	for _, r := range value {
		if r < 0x21 || r > 0x7e {
			return ""
		}
	}
	return value
}

// WithRequest stores the correlation IDs on the context.
func WithRequest(ctx context.Context, request Request) context.Context {
	return context.WithValue(ctx, requestKey, request)
}

// WithRequestID replaces only the request ID, keeping any trace already on
// the context (a queued analysis job is identified by its job ID).
func WithRequestID(ctx context.Context, id string) context.Context {
	request, ok := RequestFrom(ctx)
	if !ok {
		request = Request{TraceID: NewID()}
	}
	request.ID = sanitizeID(id)
	return WithRequest(ctx, request)
}

// RequestFrom returns the correlation IDs on the context, if any.
func RequestFrom(ctx context.Context) (Request, bool) {
	if ctx == nil {
		return Request{}, false
	}
	request, ok := ctx.Value(requestKey).(Request)
	return request, ok
}

// OutgoingTraceHeader renders the context's trace for a downstream call.
func OutgoingTraceHeader(ctx context.Context) string {
	request, ok := RequestFrom(ctx)
	if !ok || request.TraceID == "" {
		return ""
	}
	span := request.SpanID
	if span == "" {
		span = "1"
	}
	options := "0"
	if request.Sampled {
		options = "1"
	}
	return fmt.Sprintf("%s/%s;o=%s", request.TraceID, span, options)
}

var (
	outputMu sync.Mutex
	output   io.Writer = os.Stdout
	project  string
)

// Configure sets the project whose Cloud Trace the log trace field names.
func Configure(projectID string) {
	outputMu.Lock()
	defer outputMu.Unlock()
	project = strings.TrimSpace(projectID)
}

// SetOutput redirects structured logs; nil restores stdout. Tests use it.
func SetOutput(w io.Writer) {
	outputMu.Lock()
	defer outputMu.Unlock()
	if w == nil {
		w = os.Stdout
	}
	output = w
}

// Severity names the Cloud Logging levels this service uses.
type Severity string

const (
	Info    Severity = "INFO"
	Warning Severity = "WARNING"
	Error   Severity = "ERROR"
)

// Event writes one structured log entry. Fields become top-level JSON keys,
// which log-based metrics and filters can address as jsonPayload.<key>.
func Event(ctx context.Context, severity Severity, message string, fields map[string]any) {
	entry := make(map[string]any, len(fields)+6)
	for key, value := range fields {
		if err, ok := value.(error); ok {
			value = err.Error()
		}
		entry[key] = value
	}
	outputMu.Lock()
	defer outputMu.Unlock()
	entry["severity"] = string(severity)
	entry["message"] = message
	entry["time"] = time.Now().UTC().Format(time.RFC3339Nano)
	if request, ok := RequestFrom(ctx); ok {
		if request.ID != "" {
			entry["request_id"] = request.ID
		}
		if request.TraceID != "" && project != "" {
			entry["logging.googleapis.com/trace"] = "projects/" + project + "/traces/" + request.TraceID
			// The header carries the span in decimal; Cloud Logging wants 16 hex digits.
			if span, err := strconv.ParseUint(request.SpanID, 10, 64); err == nil && span != 0 {
				entry["logging.googleapis.com/spanId"] = fmt.Sprintf("%016x", span)
			}
			entry["logging.googleapis.com/trace_sampled"] = request.Sampled
		}
	}
	line, err := json.Marshal(entry)
	if err != nil {
		line, _ = json.Marshal(map[string]string{"severity": string(severity), "message": message})
	}
	_, _ = output.Write(append(line, '\n'))
}

// lineWriter turns the standard library's formatted log lines into structured
// entries of one severity, so the many existing Printf call sites become
// structured without being rewritten.
type lineWriter struct{ severity Severity }

func (w lineWriter) Write(p []byte) (int, error) {
	text := strings.TrimRight(string(p), "\n")
	fields := map[string]any{}
	// log.Lshortfile prefixes "file.go:123: ".
	if file, rest, found := strings.Cut(text, ": "); found {
		if name, lineNumber, ok := strings.Cut(file, ":"); ok && strings.HasSuffix(name, ".go") {
			if n, err := strconv.Atoi(lineNumber); err == nil {
				fields["logging.googleapis.com/sourceLocation"] = map[string]any{"file": name, "line": strconv.Itoa(n)}
				text = rest
			}
		}
	}
	Event(context.Background(), w.severity, text, fields)
	return len(p), nil
}

// NewStdLogger returns a *log.Logger whose lines are written as structured
// entries at the given severity.
func NewStdLogger(severity Severity) *log.Logger {
	return log.New(lineWriter{severity: severity}, "", log.Lshortfile)
}
