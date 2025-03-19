// pkg/core/runnable/parallel.go

package runnable

import (
	"context"
	"fmt"
	"reflect"
	"sync"
)

// RunnableParallel runs multiple runnables in parallel on the same input
// and returns a map of their outputs.
type RunnableParallel[I Input, O map[string]interface{}] struct {
	runnables map[string]interface{}
}

// NewRunnableParallel creates a new RunnableParallel with the given runnables.
func NewRunnableParallel[I Input, O map[string]interface{}](runnables map[string]interface{}) (*RunnableParallel[I, O], error) {
	if len(runnables) == 0 {
		return nil, fmt.Errorf("parallel must have at least one runnable")
	}

	// Validate that all runnables accept I as input
	for key, runnable := range runnables {
		rType := reflect.TypeOf(runnable)
		
		// Check if it implements Runnable[I, _]
		if !rType.Implements(reflect.TypeOf((*Runnable[I, interface{}])(nil)).Elem()) {
			return nil, fmt.Errorf("runnable %s must implement Runnable[I, _]", key)
		}
	}

	return &RunnableParallel[I, O]{
		runnables: runnables,
	}, nil
}

// MustNewRunnableParallel creates a new RunnableParallel and panics if there's an error.
func MustNewRunnableParallel[I Input, O map[string]interface{}](runnables map[string]interface{}) *RunnableParallel[I, O] {
	parallel, err := NewRunnableParallel[I, O](runnables)
	if err != nil {
		panic(err)
	}
	return parallel
}

// Run implements the Runnable interface.
func (p *RunnableParallel[I, O]) Run(ctx context.Context, input I) (O, error) {
	return p.RunWithConfig(ctx, input, RunConfig{})
}

// RunWithConfig implements the Runnable interface.
func (p *RunnableParallel[I, O]) RunWithConfig(ctx context.Context, input I, config RunConfig) (O, error) {
	// Create a map to store the results
	results := make(map[string]interface{})
	
	// Create a wait group to wait for all goroutines to finish
	var wg sync.WaitGroup
	wg.Add(len(p.runnables))
	
	// Create a mutex to protect the results map
	var mu sync.Mutex
	
	// Create a channel to collect errors
	errCh := make(chan error, len(p.runnables))
	
	// Run each runnable in parallel
	for key, runnable := range p.runnables {
		go func(key string, runnable interface{}) {
			defer wg.Done()
			
			// Use reflection to call the RunWithConfig method
			val := reflect.ValueOf(runnable)
			method := val.MethodByName("RunWithConfig")
			
			if !method.IsValid() {
				errCh <- fmt.Errorf("runnable %s does not have a RunWithConfig method", key)
				return
			}
			
			// Call the method
			methodResults := method.Call([]reflect.Value{
				reflect.ValueOf(ctx),
				reflect.ValueOf(input),
				reflect.ValueOf(config),
			})
			
			// Check for errors
			if !methodResults[1].IsNil() {
				errCh <- methodResults[1].Interface().(error)
				return
			}
			
			// Store the result
			mu.Lock()
			results[key] = methodResults[0].Interface()
			mu.Unlock()
		}(key, runnable)
	}
	
	// Wait for all goroutines to finish
	wg.Wait()
	close(errCh)
	
	// Check for errors
	for err := range errCh {
		if err != nil {
			var zeroO O
			return zeroO, err
		}
	}
	
	// Convert the results to the expected output type
	output, ok := interface{}(results).(O)
	if !ok {
		var zeroO O
		return zeroO, fmt.Errorf("output type does not match expected type")
	}
	
	return output, nil
}

// Stream implements the Runnable interface.
func (p *RunnableParallel[I, O]) Stream(ctx context.Context, input I) (<-chan StreamingChunk[O], error) {
	return p.StreamWithConfig(ctx, input, RunConfig{})
}

// StreamWithConfig implements the Runnable interface.
func (p *RunnableParallel[I, O]) StreamWithConfig(ctx context.Context, input I, config RunConfig) (<-chan StreamingChunk[O], error) {
	// For simplicity, we'll just run the parallel operation and stream the final result
	// A more advanced implementation would merge streams from all runnables
	
	outputCh := make(chan StreamingChunk[O])
	
	go func() {
		defer close(outputCh)
		
		result, err := p.RunWithConfig(ctx, input, config)
		if err != nil {
			// Send error as metadata
			outputCh <- StreamingChunk[O]{
				Data:  *new(O), // Zero value
				Index: 0,
				Final: true,
				Meta: map[string]interface{}{
					"error": err.Error(),
				},
			}
			return
		}
		
		// Send the result as a single chunk
		outputCh <- StreamingChunk[O]{
			Data:  result,
			Index: 0,
			Final: true,
		}
	}()
	
	return outputCh, nil
}

// GetInputSchema implements the Runnable interface.
func (p *RunnableParallel[I, O]) GetInputSchema() map[string]interface{} {
	// Combine input schemas from all runnables
	schemas := make(map[string]interface{})
	
	for key, runnable := range p.runnables {
		val := reflect.ValueOf(runnable)
		method := val.MethodByName("GetInputSchema")
		
		if !method.IsValid() {
			schemas[key] = map[string]interface{}{
				"type": "any",
			}
			continue
		}
		
		results := method.Call(nil)
		if len(results) != 1 {
			schemas[key] = map[string]interface{}{
				"type": "any",
			}
			continue
		}
		
		schema, ok := results[0].Interface().(map[string]interface{})
		if !ok {
			schemas[key] = map[string]interface{}{
				"type": "any",
			}
			continue
		}
		
		schemas[key] = schema
	}
	
	return map[string]interface{}{
		"type": "object",
		"properties": schemas,
		"description": "Input for all parallel runnables",
	}
}

// GetOutputSchema implements the Runnable interface.
func (p *RunnableParallel[I, O]) GetOutputSchema() map[string]interface{} {
	// Combine output schemas from all runnables
	schemas := make(map[string]interface{})
	
	for key, runnable := range p.runnables {
		val := reflect.ValueOf(runnable)
		method := val.MethodByName("GetOutputSchema")
		
		if !method.IsValid() {
			schemas[key] = map[string]interface{}{
				"type": "any",
			}
			continue
		}
		
		results := method.Call(nil)
		if len(results) != 1 {
			schemas[key] = map[string]interface{}{
				"type": "any",
			}
			continue
		}
		
		schema, ok := results[0].Interface().(map[string]interface{})
		if !ok {
			schemas[key] = map[string]interface{}{
				"type": "any",
			}
			continue
		}
		
		schemas[key] = schema
	}
	
	return map[string]interface{}{
		"type": "object",
		"properties": schemas,
		"description": "Output from all parallel runnables",
	}
}
