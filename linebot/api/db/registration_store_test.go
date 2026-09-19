package db

import (
	"context"
	"errors"
	"net"
	"sync"
	"testing"
	"time"

	"cloud.google.com/go/firestore"
	pb "cloud.google.com/go/firestore/apiv1/firestorepb"
	"google.golang.org/api/option"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/emptypb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// A local protocol fake, not a production database or a credentialed live test.
type registrationFirestore struct {
	pb.UnimplementedFirestoreServer
	mu       sync.Mutex
	document *pb.Document
}

func (s *registrationFirestore) BeginTransaction(context.Context, *pb.BeginTransactionRequest) (*pb.BeginTransactionResponse, error) {
	return &pb.BeginTransactionResponse{Transaction: []byte("local-test")}, nil
}
func (s *registrationFirestore) Rollback(context.Context, *pb.RollbackRequest) (*emptypb.Empty, error) {
	return &emptypb.Empty{}, nil
}
func (s *registrationFirestore) BatchGetDocuments(request *pb.BatchGetDocumentsRequest, stream pb.Firestore_BatchGetDocumentsServer) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, name := range request.Documents {
		response := &pb.BatchGetDocumentsResponse{ReadTime: timestamppb.Now()}
		if s.document != nil && s.document.Name == name {
			response.Result = &pb.BatchGetDocumentsResponse_Found{Found: proto.Clone(s.document).(*pb.Document)}
		} else {
			response.Result = &pb.BatchGetDocumentsResponse_Missing{Missing: name}
		}
		if err := stream.Send(response); err != nil {
			return err
		}
	}
	return nil
}
func (s *registrationFirestore) Commit(_ context.Context, request *pb.CommitRequest) (*pb.CommitResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := timestamppb.Now()
	result := &pb.CommitResponse{CommitTime: now}
	for _, write := range request.Writes {
		update := write.GetUpdate()
		if update == nil {
			return nil, errors.New("unexpected write operation")
		}
		if s.document == nil || write.UpdateMask == nil {
			s.document = proto.Clone(update).(*pb.Document)
		} else {
			for _, field := range write.UpdateMask.FieldPaths {
				if value, ok := update.Fields[field]; ok {
					s.document.Fields[field] = value
				} else {
					delete(s.document.Fields, field)
				}
			}
		}
		item := &pb.WriteResult{UpdateTime: now}
		for _, transform := range write.UpdateTransforms {
			value := &pb.Value{ValueType: &pb.Value_TimestampValue{TimestampValue: now}}
			s.document.Fields[transform.FieldPath] = value
			item.TransformResults = append(item.TransformResults, value)
		}
		s.document.CreateTime = now
		s.document.UpdateTime = now
		result.WriteResults = append(result.WriteResults, item)
	}
	return result, nil
}

// newRegistrationTestClient starts an in-memory Firestore and hands back a
// client pointed at it.
func newRegistrationTestClient(t *testing.T) *FirestoreClient {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	listener := bufconn.Listen(1024 * 1024)
	server := grpc.NewServer()
	pb.RegisterFirestoreServer(server, &registrationFirestore{})
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(server.Stop)
	connection, err := grpc.DialContext(ctx, "passthrough:///registration-test",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = connection.Close() })
	transport, err := firestore.NewClient(ctx, "registration-test", option.WithGRPCConn(connection), option.WithoutAuthentication())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = transport.Close() })
	return &FirestoreClient{Ctx: &ctx, Client: transport, Data: transport.Collection("variant_users")}
}

// The learner fills the form in themselves, so a typo in their own name has to
// be fixable, and an unrelated write must not erase what they saved.
func TestProfileSaveRegistersThenAllowsEdits(t *testing.T) {
	client := newRegistrationTestClient(t)
	stale := &UserData{ID: "U1", Name: "LINE display name", Portfolio: Portfolios{
		Serve: map[string]Work{"attempt": {Reflection: "keep this note"}}}}
	if _, err := client.Data.Doc("U1").Set(*client.Ctx, stale); err != nil {
		t.Fatal(err)
	}

	saved, err := client.SaveExperimentRegistration("U1", ExperimentRegistration{ExperimentNumber: "001", RealName: "王小明"})
	if err != nil {
		t.Fatal(err)
	}
	if !saved.HasExperimentRegistration() || saved.RealName != "王小明" || saved.ExperimentNumber != "001" {
		t.Fatalf("registration not persisted: %+v", saved)
	}
	joined := saved.RegistrationCompletedAt

	edited, err := client.SaveExperimentRegistration("U1", ExperimentRegistration{ExperimentNumber: "EG02", RealName: "王小美"})
	if err != nil {
		t.Fatalf("edit rejected: %v", err)
	}
	if edited.RealName != "王小美" || edited.ExperimentNumber != "EG02" {
		t.Fatalf("edit not persisted: %+v", edited)
	}
	if !edited.RegistrationCompletedAt.Equal(joined) {
		t.Error("editing details moved the time the learner joined the experiment")
	}
	if edited.Name != "LINE display name" || edited.Portfolio.Serve["attempt"].Reflection != "keep this note" {
		t.Error("save overwrote unrelated fields")
	}

	// A write built from a record read before registration must not carry the
	// empty fields back over it.
	if err := client.UpdateUserHandedness(stale, Left); err != nil {
		t.Fatal(err)
	}
	after, err := client.GetUserData("U1")
	if err != nil || !after.HasExperimentRegistration() {
		t.Fatalf("stale update erased the registration: %v", err)
	}
	if after.Handedness != Left {
		t.Error("handedness was not saved")
	}

	if _, err := client.SaveExperimentRegistration("U1", ExperimentRegistration{ExperimentNumber: "abc", RealName: "王"}); !errors.Is(err, ErrRegistrationFormat) {
		t.Fatalf("invalid details were not rejected: %v", err)
	}
}
