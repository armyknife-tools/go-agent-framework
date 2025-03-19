// pkg/llms/openai/openai.go

package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/armyknife-tools/go-agent-framework/pkg/core/llm"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/runnable"
)

const (
	defaultBaseURL = "https://api.openai.com/v1"
	defaultModel   = "gpt-3.5-turbo-instruct"
)

// OpenAILLM implements a language model using OpenAI's API.
type OpenAILLM struct {
	APIKey       string
	BaseURL      string
	Model        string
	Temperature  float64
	MaxTokens    int
	TopP         float64
	HttpClient   *http.Client
	Organization string
}

// NewOpenAILLM creates a new OpenAILLM with the given API key.
func NewOpenAILLM(apiKey string, opts ...Option) *OpenAILLM {
	llm := &OpenAILLM{
		APIKey:      apiKey,
		BaseURL:     defaultBaseURL,
		Model:       defaultModel,
		Temperature: 0.7,
		MaxTokens:   1000,
		TopP:        1.0,
		HttpClient:  &http.Client{},
	}

	// Apply options
	for _, opt := range opts {
		opt(llm)
	}

	return llm
}

// Option is a function that configures an OpenAILLM.
type Option func(*OpenAILLM)

// WithModel sets the model.
func WithModel(model string) Option {
	return func(o *OpenAILLM) {
		o.Model = model
	}
}

// WithTemperature sets the temperature.
func WithTemperature(temperature float64) Option {
	return func(o *OpenAILLM) {
		o.Temperature = temperature
	}
}

// WithMaxTokens sets the maximum number of tokens.
func WithMaxTokens(maxTokens int) Option {
	return func(o *OpenAILLM) {
		o.MaxTokens = maxTokens
	}
}

// WithTopP sets the top_p value.
func WithTopP(topP float64) Option {
	return func(o *OpenAILLM) {
		o.TopP = topP
	}
}

// WithBaseURL sets the base URL.
func WithBaseURL(baseURL string) Option {
	return func(o *OpenAILLM) {
		o.BaseURL = baseURL
	}
}

// WithHttpClient sets the HTTP client.
func WithHttpClient(client *http.Client) Option {
	return func(o *OpenAILLM) {
		o.HttpClient = client
	}
}

// WithOrganization sets the OpenAI organization.
func WithOrganization(org string) Option {
	return func(o *OpenAILLM) {
		o.Organization = org
	}
}

// completionRequest represents a request to the OpenAI completion API.
type completionRequest struct {
	Model            string   `json:"model"`
	Prompt           string   `json:"prompt"`
	Temperature      float64  `json:"temperature"`
	MaxTokens        int      `json:"max_tokens"`
	TopP             float64  `json:"top_p,omitempty"`
	FrequencyPenalty float64  `json:"frequency_penalty,omitempty"`
	PresencePenalty  float64  `json:"presence_penalty,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	Stream           bool     `json:"stream,omitempty"`
}

// completionResponse represents a response from the OpenAI completion API.
type completionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Text        string `json:"text"`
		Index       int    `json:"index"`
		LogProbs    any    `json:"logprobs"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// Generate implements the BaseLanguageModel interface.
func (o *OpenAILLM) Generate(ctx context.Context, prompts []llm.Prompt, options llm.ModelOptions) (llm.LLMResult, error) {
	result := llm.LLMResult{
		Generations: make([][]llm.Generation, len(prompts)),
		ModelInfo: map[string]interface{}{
			"model": o.Model,
		},
	}

	for i, prompt := range prompts {
		var promptStr string
		switch p := prompt.(type) {
		case string:
			promptStr = p
		case llm.StringPrompt:
			promptStr = string(p)
		default:
			return result, fmt.Errorf("unsupported prompt type: %T", prompt)
		}

		// Apply options
		temp := o.Temperature
		if options.Temperature != 0 {
			temp = options.Temperature
		}

		maxTokens := o.MaxTokens
		if options.MaxTokens != 0 {
			maxTokens = options.MaxTokens
		}

		topP := o.TopP
		if options.TopP != 0 {
			topP = options.TopP
		}

		model := o.Model
		if options.ModelName != "" {
			model = options.ModelName
		}

		// Create request
		reqBody := completionRequest{
			Model:            model,
			Prompt:           promptStr,
			Temperature:      temp,
			MaxTokens:        maxTokens,
			TopP:             topP,
			FrequencyPenalty: options.FrequencyPenalty,
			PresencePenalty:  options.PresencePenalty,
			Stop:             options.StopSequences,
		}

		// Serialize request
		jsonData, err := json.Marshal(reqBody)
		if err != nil {
			return result, fmt.Errorf("error marshaling request: %w", err)
		}

		// Create HTTP request
		req, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/completions", o.BaseURL), bytes.NewBuffer(jsonData))
		if err != nil {
			return result, fmt.Errorf("error creating request: %w", err)
		}

		// Set headers
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", o.APIKey))
		if o.Organization != "" {
			req.Header.Set("OpenAI-Organization", o.Organization)
		}

		// Make the request
		resp, err := o.HttpClient.Do(req)
		if err != nil {
			return result, fmt.Errorf("error making request: %w", err)
		}
		defer resp.Body.Close()

		// Check for error response
		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			return result, fmt.Errorf("API error (status %d): %s", resp.StatusCode, body)
		}

		// Parse response
		var compResp completionResponse
		if err := json.NewDecoder(resp.Body).Decode(&compResp); err != nil {
			return result, fmt.Errorf("error decoding response: %w", err)
		}

		// Extract generations
		var generations []llm.Generation
		for _, choice := range compResp.Choices {
			generations = append(generations, llm.Generation{
				Text: strings.TrimSpace(choice.Text),
				TokenUsage: llm.TokenUsage{
					PromptTokens:     compResp.Usage.PromptTokens,
					CompletionTokens: compResp.Usage.CompletionTokens,
					TotalTokens:      compResp.Usage.TotalTokens,
				},
				Metadata: map[string]interface{}{
					"finish_reason": choice.FinishReason,
				},
			})
		}

		result.Generations[i] = generations
	}

	return result, nil
}

// GeneratePrompt implements the BaseLanguageModel interface.
func (o *OpenAILLM) GeneratePrompt(ctx context.Context, prompt llm.Prompt, options llm.ModelOptions) ([]llm.Generation, error) {
	result, err := o.Generate(ctx, []llm.Prompt{prompt}, options)
	if err != nil {
		return nil, err
	}
	
	if len(result.Generations) == 0 || len(result.Generations[0]) == 0 {
		return nil, fmt.Errorf("no generations returned")
	}
	
	return result.Generations[0], nil
}

// Run implements the Runnable interface.
func (o *OpenAILLM) Run(ctx context.Context, input llm.Prompt) ([]llm.Generation, error) {
	return o.GeneratePrompt(ctx, input, llm.ModelOptions{})
}

// RunWithConfig implements the Runnable interface.
func (o *OpenAILLM) RunWithConfig(ctx context.Context, input llm.Prompt, config runnable.RunConfig) ([]llm.Generation, error) {
	return o.GeneratePrompt(ctx, input, llm.ModelOptions{
		Temperature: o.Temperature,
		MaxTokens:   o.MaxTokens,
		TopP:        o.TopP,
	})
}

// Stream implements the BaseLanguageModel interface.
func (o *OpenAILLM) Stream(ctx context.Context, prompt llm.Prompt, options llm.ModelOptions) (<-chan llm.Generation, error) {
	// Implementation omitted for brevity - would use SSE for streaming
	// In a real implementation, this would establish an SSE connection and parse the chunks
	return nil, fmt.Errorf("streaming not implemented")
}

// GetInputSchema implements the Runnable interface.
func (o *OpenAILLM) GetInputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "string",
		"description": "The prompt to generate completions for",
	}
}

// GetOutputSchema implements the Runnable interface.
func (o *OpenAILLM) GetOutputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "array",
		"items": map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"text": map[string]interface{}{
					"type": "string",
					"description": "The generated completion text",
				},
				"token_usage": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"prompt_tokens": map[string]interface{}{"type": "integer"},
						"completion_tokens": map[string]interface{}{"type": "integer"},
						"total_tokens": map[string]interface{}{"type": "integer"},
					},
				},
			},
		},
	}
}

// Stream implements the Runnable interface.
func (o *OpenAILLM) StreamWithConfig(ctx context.Context, input llm.Prompt, config runnable.RunConfig) (<-chan runnable.StreamingChunk[[]llm.Generation], error) {
	// Implementation omitted for brevity
	return nil, fmt.Errorf("streaming not implemented")
}

// ModelName returns the name of the model.
func (o *OpenAILLM) ModelName() string {
	return o.Model
}

// ModelType returns the type of the model.
func (o *OpenAILLM) ModelType() string {
	return "llm"
}
