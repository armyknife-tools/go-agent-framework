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
	runnable.Runnable[Prompt, []Generation]
}

// StreamingLanguageModel extends LanguageModel with streaming capabilities.
type StreamingLanguageModel interface {
	LanguageModel
	// StreamingGenerate returns a stream of tokens.
	StreamingGenerate(ctx context.Context, prompt Prompt, options ModelOptions) (<-chan runnable.StreamingChunk[string], error)
}
