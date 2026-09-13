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
func TestRegistrationPersistsAndSurvivesStalePortfolioWrite(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	listener := bufconn.Listen(1024 * 1024)
	server := grpc.NewServer()
	pb.RegisterFirestoreServer(server, &registrationFirestore{})
	go func() { _ = server.Serve(listener) }()
	defer server.Stop()
	connection, err := grpc.DialContext(ctx, "passthrough:///registration-test",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer connection.Close()
	transport, err := firestore.NewClient(ctx, "registration-test", option.WithGRPCConn(connection), option.WithoutAuthentication())
	if err != nil {
		t.Fatal(err)
	}
	defer transport.Close()
	client := &FirestoreClient{Ctx: &ctx, Client: transport, Data: transport.Collection("variant_users")}
	stale := &UserData{ID: "U1", Name: "LINE display name", Portfolio: Portfolios{
		Serve: map[string]Work{"attempt": {Reflection: "keep this note"}}}}
	if _, err := client.Data.Doc("U1").Set(ctx, stale); err != nil {
		t.Fatal(err)
	}
	registration := ExperimentRegistration{ExperimentNumber: "001", RealName: "王小明"}
	saved, err := client.RegisterExperiment("U1", registration)
	if err != nil {
		t.Fatal(err)
	}
	if !saved.HasExperimentRegistration() || saved.RealName != "王小明" || saved.ExperimentNumber != "001" {
		t.Fatalf("registration not persisted: %+v", saved)
	}
	if saved.Name != "LINE display name" || saved.Portfolio.Serve["attempt"].Reflection != "keep this note" {
		t.Fatal("registration overwrote unrelated fields")
	}
	if err := client.UpdateUserHandedness(stale, Left); err != nil {
		t.Fatal(err)
	}
	saved, err = client.GetUserData("U1")
	if err != nil || !saved.HasExperimentRegistration() {
		t.Fatalf("stale update erased registration: %v", err)
	}
	if _, err := client.RegisterExperiment("U1", registration); err != nil {
		t.Fatalf("retry failed: %v", err)
	}
	if _, err := client.RegisterExperiment("U1", ExperimentRegistration{ExperimentNumber: "002", RealName: "李小明"}); !errors.Is(err, ErrRegistrationLocked) {
		t.Fatalf("conflicting registration was not rejected: %v", err)
	}
}
