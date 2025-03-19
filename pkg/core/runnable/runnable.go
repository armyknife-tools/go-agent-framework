// pkg/core/runnable/runnable.go

package runnable

import (
	"context"
)

// Input represents the input to a Runnable
type Input interface{}

// Output represents the output from a Runnable
type Output interface{}

// Runnable is the core interface for all components in LangChain Go.
// It defines a common pattern for executing operations with different input and output types.
type Runnable[I Input, O Output] interface {
	// Run executes the runnable with the given input and returns an output.
	// It accepts a context for cancellation, timeouts, and passing values.
	Run(ctx context.Context, input I) (O, error)

	// RunWithConfig executes the runnable with the given input and configuration.
	RunWithConfig(ctx context.Context, input I, config RunConfig) (O, error)

	// Stream executes the runnable and returns a channel of output chunks.
	Stream(ctx context.Context, input I) (<-chan StreamingChunk[O], error)

	// StreamWithConfig executes the runnable with streaming and custom configuration.
	StreamWithConfig(ctx context.Context, input I, config RunConfig) (<-chan StreamingChunk[O], error)

	// GetInputSchema returns the JSON schema for the input type.
	GetInputSchema() map[string]interface{}

	// GetOutputSchema returns the JSON schema for the output type.
	GetOutputSchema() map[string]interface{}
}

// RunConfig contains configuration for a single run of a Runnable.
type RunConfig struct {
	// Configuration fields common to all runnables
	MaxRetries   int                    // Maximum number of retries on failure
	Timeout      int                    // Timeout in seconds
	Tags         []string               // Tags for the run (for logging/tracing)
	Metadata     map[string]interface{} // Additional metadata
	Callbacks    []Callback             // Callbacks to invoke during execution
	RetryBackoff BackoffStrategy        // Strategy for retry backoff
}

// BackoffStrategy defines the strategy for retry backoff.
type BackoffStrategy interface {
	// NextBackoff returns the next backoff duration based on the attempt number.
	NextBackoff(attempt int) int
}

// StreamingChunk represents a chunk of output in a streaming response.
type StreamingChunk[T any] struct {
	Data  T                      // The chunk data
	Index int                    // Index of the chunk in the stream
	Final bool                   // Whether this is the final chunk
	Meta  map[string]interface{} // Additional metadata about the chunk
}

// Callback defines the interface for callbacks that can be invoked during runnable execution.
type Callback interface {
	// OnRunStart is called when a runnable starts execution.
	OnRunStart(ctx context.Context, input Input) error

	// OnRunFinish is called when a runnable finishes execution.
	OnRunFinish(ctx context.Context, input Input, output Output, err error) error

	// OnStreamChunk is called for each chunk in a streaming response.
	OnStreamChunk(ctx context.Context, chunk interface{}) error
}
