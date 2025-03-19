// pkg/core/llm/language_model.go

package llm

import (
	"context"

	"github.com/armyknife-tools/go-agent-framework/pkg/core/runnable"
)

// Prompt represents input to a language model.
// It can be a simple string or a more complex prompt structure.
type Prompt interface{}

// StringPrompt is a simple string prompt.
type StringPrompt string

// Generation represents a single text generation from a language model.
type Generation struct {
	Text       string                 `json:"text"`
	TokenUsage TokenUsage             `json:"token_usage,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// TokenUsage tracks token counts for a model call.
type TokenUsage struct {
	PromptTokens      int `json:"prompt_tokens"`
	CompletionTokens  int `json:"completion_tokens"`
	TotalTokens       int `json:"total_tokens"`
}

// LLMResult contains the complete response from a language model.
type LLMResult struct {
	Generations [][]Generation        `json:"generations"`
	ModelInfo   map[string]interface{} `json:"model_info,omitempty"`
}

// ModelOptions defines common options for language model calls.
type ModelOptions struct {
	Temperature      float64            `json:"temperature,omitempty"`
	MaxTokens        int                `json:"max_tokens,omitempty"`
	StopSequences    []string           `json:"stop_sequences,omitempty"`
	TopP             float64            `json:"top_p,omitempty"`
	FrequencyPenalty float64            `json:"frequency_penalty,omitempty"`
	PresencePenalty  float64            `json:"presence_penalty,omitempty"`
	ModelName        string             `json:"model_name,omitempty"`
	Timeout          int                `json:"timeout,omitempty"`
	Cache            bool               `json:"cache,omitempty"`
	APIKey           string             `json:"api_key,omitempty"`
	Extra            map[string]interface{} `json:"extra,omitempty"`
}

// BaseLanguageModel is the interface that all language models must implement.
type BaseLanguageModel interface {
	// Generate produces completions for the given prompts.
	Generate(ctx context.Context, prompts []Prompt, options ModelOptions) (LLMResult, error)
	
	// GeneratePrompt is a convenience method for generating completions for a single prompt.
	GeneratePrompt(ctx context.Context, prompt Prompt, options ModelOptions) ([]Generation, error)
	
	// Stream returns completions as they're generated.
	Stream(ctx context.Context, prompt Prompt, options ModelOptions) (<-chan Generation, error)
	
	// ModelName returns the name of the model.
	ModelName() string
	
	// ModelType returns the type of the model (e.g., "llm", "chat").
	ModelType() string
}

// LanguageModel is a Runnable implementation of BaseLanguageModel.
type LanguageModel interface {
	BaseLanguageModel
}

// StreamingLanguageModel extends LanguageModel with streaming capabilities.
type StreamingLanguageModel interface {
	LanguageModel
	// StreamingGenerate returns a stream of tokens.
	StreamingGenerate(ctx context.Context, prompt Prompt, options ModelOptions) (<-chan runnable.StreamingChunk[string], error)
}

// RunnableLanguageModel is a helper type that adapts a BaseLanguageModel to the Runnable interface.
type RunnableLanguageModel struct {
	model BaseLanguageModel
}

// NewRunnableLanguageModel creates a new RunnableLanguageModel from a BaseLanguageModel.
func NewRunnableLanguageModel(model BaseLanguageModel) *RunnableLanguageModel {
	return &RunnableLanguageModel{model: model}
}

// Run implements runnable.Runnable.Run
func (r *RunnableLanguageModel) Run(ctx context.Context, input Prompt) ([]Generation, error) {
	return r.model.GeneratePrompt(ctx, input, ModelOptions{})
}

// RunWithConfig implements runnable.Runnable.RunWithConfig
func (r *RunnableLanguageModel) RunWithConfig(ctx context.Context, input Prompt, config runnable.RunConfig) ([]Generation, error) {
	// Convert runnable config to model options if needed
	options := ModelOptions{}
	if config.Timeout > 0 {
		options.Timeout = config.Timeout
	}
	return r.model.GeneratePrompt(ctx, input, options)
}

// Stream implements runnable.Runnable.Stream
func (r *RunnableLanguageModel) Stream(ctx context.Context, input Prompt) (<-chan runnable.StreamingChunk[[]Generation], error) {
	genChan, err := r.model.Stream(ctx, input, ModelOptions{})
	if err != nil {
		return nil, err
	}
	
	// Convert the generation channel to a streaming chunk channel
	resultChan := make(chan runnable.StreamingChunk[[]Generation])
	go func() {
		defer close(resultChan)
		index := 0
		for gen := range genChan {
			resultChan <- runnable.StreamingChunk[[]Generation]{
				Data:  []Generation{gen},
				Index: index,
				Final: false,
			}
			index++
		}
		// Send final chunk
		resultChan <- runnable.StreamingChunk[[]Generation]{
			Data:  []Generation{},
			Index: index,
			Final: true,
		}
	}()
	
	return resultChan, nil
}

// StreamWithConfig implements runnable.Runnable.StreamWithConfig
func (r *RunnableLanguageModel) StreamWithConfig(ctx context.Context, input Prompt, config runnable.RunConfig) (<-chan runnable.StreamingChunk[[]Generation], error) {
	// Convert runnable config to model options if needed
	options := ModelOptions{}
	if config.Timeout > 0 {
		options.Timeout = config.Timeout
	}
	
	genChan, err := r.model.Stream(ctx, input, options)
	if err != nil {
		return nil, err
	}
	
	// Convert the generation channel to a streaming chunk channel
	resultChan := make(chan runnable.StreamingChunk[[]Generation])
	go func() {
		defer close(resultChan)
		index := 0
		for gen := range genChan {
			resultChan <- runnable.StreamingChunk[[]Generation]{
				Data:  []Generation{gen},
				Index: index,
				Final: false,
			}
			index++
		}
		// Send final chunk
		resultChan <- runnable.StreamingChunk[[]Generation]{
			Data:  []Generation{},
			Index: index,
			Final: true,
		}
	}()
	
	return resultChan, nil
}

// GetInputSchema implements runnable.Runnable.GetInputSchema
func (r *RunnableLanguageModel) GetInputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"prompt": map[string]interface{}{
				"type": "string",
				"description": "The prompt to send to the language model",
			},
		},
	}
}

// GetOutputSchema implements runnable.Runnable.GetOutputSchema
func (r *RunnableLanguageModel) GetOutputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "array",
		"items": map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"text": map[string]interface{}{
					"type": "string",
					"description": "The generated text",
				},
			},
		},
	}
}

// ToRunnable converts a LanguageModel to a Runnable.
// This is a convenience function for using a LanguageModel in a runnable chain.
func ToRunnable(model LanguageModel) runnable.Runnable[Prompt, []Generation] {
	return NewRunnableLanguageModel(model)
}
