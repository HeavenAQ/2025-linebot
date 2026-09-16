package utils_test

import (
	"testing"

	"github.com/HeavenAQ/nstc-linebot-2025/utils"
)

func TestRandomInt(t *testing.T) {
	t.Parallel()

	type args struct {
		min int64
		max int64
	}

	tests := []struct {
		want func(val int64) bool
		name string
		args args
	}{
		{
			name: "Test RandomInt",
			args: args{
				min: 0,
				max: 100,
			},
			want: func(val int64) bool {
				return val >= 0 && val <= 100
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if got := utils.RandomInt(tt.args.min, tt.args.max); !tt.want(got) {
				t.Errorf("RandomInt() = %v, want 0 <= val <= 100", got)
			}
		})
	}
}

func TestRandomString(t *testing.T) {
	t.Parallel()

	type args struct {
		n int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{
			name: "Test RandomString",
			args: args{
				n: 10,
			},
			want: 10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if got := utils.RandomAlphabetString(tt.args.n); len(got) != tt.want {
				t.Errorf("RandomString() = %v, want %v", got, tt.want)
			}
		})
	}
}
