package review

import (
	"crypto/sha256"
	"crypto/subtle"
	"net"
	"net/http"
	"sort"
	"strings"
)

// AdminID is the access-code id that carries the admin role.
const AdminID = "admin"

// Role is what a caller may do.
type Role string

const (
	RoleExpert Role = "expert"
	RoleAdmin  Role = "admin"
)

// Principal is an authenticated caller.
type Principal struct {
	ID   string
	Role Role
}

type codeEntry struct {
	id   string
	hash [sha256.Size]byte
}

// Authenticator matches access codes to principals.
type Authenticator struct {
	entries []codeEntry
}

// NewAuthenticator builds an authenticator from expert_id -> code.
func NewAuthenticator(codes map[string]string) *Authenticator {
	a := &Authenticator{}
	for id, code := range codes {
		a.entries = append(a.entries, codeEntry{id: id, hash: sha256.Sum256([]byte(code))})
	}
	sort.Slice(a.entries, func(i, j int) bool { return a.entries[i].id < a.entries[j].id })
	return a
}

// Authenticate compares the code against every entry in constant time (the
// codes are hashed first so lengths do not leak) and never stops early.
func (a *Authenticator) Authenticate(code string) (Principal, bool) {
	if code == "" {
		return Principal{}, false
	}
	given := sha256.Sum256([]byte(code))
	matched := -1
	for i, e := range a.entries {
		if subtle.ConstantTimeCompare(given[:], e.hash[:]) == 1 {
			matched = i
		}
	}
	if matched < 0 {
		return Principal{}, false
	}
	id := a.entries[matched].id
	role := RoleExpert
	if id == AdminID {
		role = RoleAdmin
	}
	return Principal{ID: id, Role: role}, true
}

// ExpertIDs lists the non-admin ids, sorted.
func (a *Authenticator) ExpertIDs() []string {
	var ids []string
	for _, e := range a.entries {
		if e.id != AdminID {
			ids = append(ids, e.id)
		}
	}
	return ids
}

// ClientIP is the last X-Forwarded-For entry (appended by Cloud Run's front
// end), else the connection's remote address.
func ClientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		parts := strings.Split(xff, ",")
		if last := strings.TrimSpace(parts[len(parts)-1]); last != "" {
			return last
		}
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}
