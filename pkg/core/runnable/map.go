// pkg/core/runnable/map.go

package runnable

import (
	"context"
	"fmt"
	"reflect"
)

// InputMapper is a function that transforms the input to a different type.
type InputMapper[I, O any] func(I) (O, error)

// RunnableMap transforms the input using a mapper function before passing it to the wrapped runnable.
type RunnableMap[I, M, O any] struct {
	mapper   InputMapper[I, M]
	runnable Runnable[M, O]
}

// NewRunnableMap creates a new RunnableMap with the given mapper and runnable.
func NewRunnableMap[I, M, O any](mapper InputMapper[I, M], runnable Runnable[M, O]) *RunnableMap[I, M, O] {
	return &RunnableMap[I, M, O]{
		mapper:   mapper,
		runnable: runnable,
	}
}

// Run implements the Runnable interface.
func (m *RunnableMap[I, M, O]) Run(ctx context.Context, input I) (O, error) {
	return m.RunWithConfig(ctx, input, RunConfig{})
}

// RunWithConfig implements the Runnable interface.
func (m *RunnableMap[I, M, O]) RunWithConfig(ctx context.Context, input I, config RunConfig) (O, error) {
	// Map the input
	mappedInput, err := m.mapper(input)
	if err != nil {
		var zeroO O
		return zeroO, fmt.Errorf("error mapping input: %w", err)
	}
	
	// Run the wrapped runnable
	return m.runnable.RunWithConfig(ctx, mappedInput, config)
}

// Stream implements the Runnable interface.
func (m *RunnableMap[I, M, O]) Stream(ctx context.Context, input I) (<-chan StreamingChunk[O], error) {
	return m.StreamWithConfig(ctx, input, RunConfig{})
}

// StreamWithConfig implements the Runnable interface.
func (m *RunnableMap[I, M, O]) StreamWithConfig(ctx context.Context, input I, config RunConfig) (<-chan StreamingChunk[O], error) {
	// Map the input
	mappedInput, err := m.mapper(input)
	if err != nil {
		return nil, fmt.Errorf("error mapping input: %w", err)
	}
	
	// Stream from the wrapped runnable
	return m.runnable.StreamWithConfig(ctx, mappedInput, config)
}

// GetInputSchema implements the Runnable interface.
func (m *RunnableMap[I, M, O]) GetInputSchema() map[string]interface{} {
	// For simplicity, return a generic schema
	// A more advanced implementation would derive this from the mapper function
	return map[string]interface{}{
		"type": "any",
		"description": "Input to be mapped",
	}
}

// GetOutputSchema implements the Runnable interface.
func (m *RunnableMap[I, M, O]) GetOutputSchema() map[string]interface{} {
	return m.runnable.GetOutputSchema()
}

// RunnablePassthrough returns the input as-is.
type RunnablePassthrough[T any] struct{}

// NewRunnablePassthrough creates a new RunnablePassthrough.
func NewRunnablePassthrough[T any]() *RunnablePassthrough[T] {
	return &RunnablePassthrough[T]{}
}

// Run implements the Runnable interface.
func (p *RunnablePassthrough[T]) Run(ctx context.Context, input T) (T, error) {
	return input, nil
}

// RunWithConfig implements the Runnable interface.
func (p *RunnablePassthrough[T]) RunWithConfig(ctx context.Context, input T, config RunConfig) (T, error) {
	return input, nil
}

// Stream implements the Runnable interface.
func (p *RunnablePassthrough[T]) Stream(ctx context.Context, input T) (<-chan StreamingChunk[T], error) {
	return p.StreamWithConfig(ctx, input, RunConfig{})
}

// StreamWithConfig implements the Runnable interface.
func (p *RunnablePassthrough[T]) StreamWithConfig(ctx context.Context, input T, config RunConfig) (<-chan StreamingChunk[T], error) {
	outputCh := make(chan StreamingChunk[T], 1)
	outputCh <- StreamingChunk[T]{
		Data:  input,
		Index: 0,
		Final: true,
	}
	close(outputCh)
	return outputCh, nil
}

// GetInputSchema implements the Runnable interface.
func (p *RunnablePassthrough[T]) GetInputSchema() map[string]interface{} {
	// Derive schema from type T if possible
	t := reflect.TypeOf((*T)(nil)).Elem()
	return schemaFromType(t)
}

// GetOutputSchema implements the Runnable interface.
func (p *RunnablePassthrough[T]) GetOutputSchema() map[string]interface{} {
	return p.GetInputSchema()
}

// schemaFromType creates a JSON schema from a Go type.
func schemaFromType(t reflect.Type) map[string]interface{} {
	switch t.Kind() {
	case reflect.String:
		return map[string]interface{}{
			"type": "string",
		}
	case reflect.Bool:
		return map[string]interface{}{
			"type": "boolean",
		}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]interface{}{
			"type": "integer",
		}
	case reflect.Float32, reflect.Float64:
		return map[string]interface{}{
			"type": "number",
		}
	case reflect.Slice, reflect.Array:
		elemType := t.Elem()
		return map[string]interface{}{
			"type":  "array",
			"items": schemaFromType(elemType),
		}
	case reflect.Map:
		keyType := t.Key()
		elemType := t.Elem()
		
		if keyType.Kind() != reflect.String {
			// Only string keys are supported in JSON Schema
			return map[string]interface{}{
				"type": "object",
			}
		}
		
		return map[string]interface{}{
			"type":                 "object",
			"additionalProperties": schemaFromType(elemType),
		}
	case reflect.Struct:
		properties := make(map[string]interface{})
		required := []string{}
		
		for i := 0; i < t.NumField(); i++ {
			field := t.Field(i)
			
			// Skip unexported fields
			if field.PkgPath != "" {
				continue
			}
			
			// Get the JSON name of the field
			jsonName := field.Tag.Get("json")
			if jsonName == "" || jsonName == "-" {
				jsonName = field.Name
			}
			
			// Add the field to the properties
			properties[jsonName] = schemaFromType(field.Type)
			
			// Check if the field is required
			if !field.Type.Kind().String().HasPrefix("*") {
				required = append(required, jsonName)
			}
		}
		
		schema := map[string]interface{}{
			"type":       "object",
			"properties": properties,
		}
		
		if len(required) > 0 {
			schema["required"] = required
		}
		
		return schema
	default:
		return map[string]interface{}{
			"type": "any",
		}
	}
}

// RunnableLambda wraps a function to make it a Runnable.
type RunnableLambda[I, O any] struct {
	fn        func(context.Context, I) (O, error)
	inputType  reflect.Type
	outputType reflect.Type
}

// NewRunnableLambda creates a new RunnableLambda with the given function.
func NewRunnableLambda[I, O any](fn func(context.Context, I) (O, error)) *RunnableLambda[I, O] {
	return &RunnableLambda[I, O]{
		fn:        fn,
		inputType:  reflect.TypeOf((*I)(nil)).Elem(),
		outputType: reflect.TypeOf((*O)(nil)).Elem(),
	}
}

// Run implements the Runnable interface.
func (l *RunnableLambda[I, O]) Run(ctx context.Context, input I) (O, error) {
	return l.fn(ctx, input)
}

// RunWithConfig implements the Runnable interface.
func (l *RunnableLambda[I, O]) RunWithConfig(ctx context.Context, input I, config RunConfig) (O, error) {
	// For simplicity, we ignore the config
	return l.fn(ctx, input)
}

// Stream implements the Runnable interface.
func (l *RunnableLambda[I, O]) Stream(ctx context.Context, input I) (<-chan StreamingChunk[O], error) {
	return l.StreamWithConfig(ctx, input, RunConfig{})
}

// StreamWithConfig implements the Runnable interface.
func (l *RunnableLambda[I, O]) StreamWithConfig(ctx context.Context, input I, config RunConfig) (<-chan StreamingChunk[O], error) {
	outputCh := make(chan StreamingChunk[O], 1)
	
	go func() {
		defer close(outputCh)
		
		output, err := l.fn(ctx, input)
		if err != nil {
			// Send error as metadata
			var zeroO O
			outputCh <- StreamingChunk[O]{
				Data:  zeroO,
				Index: 0,
				Final: true,
				Meta: map[string]interface{}{
					"error": err.Error(),
				},
			}
			return
		}
		
		outputCh <- StreamingChunk[O]{
			Data:  output,
			Index: 0,
			Final: true,
		}
	}()
	
	return outputCh, nil
}

// GetInputSchema implements the Runnable interface.
func (l *RunnableLambda[I, O]) GetInputSchema() map[string]interface{} {
	return schemaFromType(l.inputType)
}

// GetOutputSchema implements the Runnable interface.
func (l *RunnableLambda[I, O]) GetOutputSchema() map[string]interface{} {
	return schemaFromType(l.outputType)
}
