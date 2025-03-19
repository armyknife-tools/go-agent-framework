// pkg/llms/anthropic/anthropic.go

package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/armyknife-tools/go-agent-framework/pkg/core/chat"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/llm"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/runnable"
)

const (
	defaultBaseURL       = "https://api.anthropic.com"
	defaultAPIVersion    = "2023-06-01"
	defaultCompletionURL = "/v1/complete"
	defaultMessagesURL   = "/v1/messages"
	defaultModel         = "claude-3-opus-20240229"
	defaultMaxTokens     = 1000
)

// AnthropicLLM implements a language model using Anthropic's API.
type AnthropicLLM struct {
	APIKey      string
	BaseURL     string
	APIVersion  string
	Model       string
	MaxTokens   int
	Temperature float64
	TopP        float64
	TopK        int
	HttpClient  *http.Client
}

// NewAnthropicLLM creates a new AnthropicLLM with the given API key.
func NewAnthropicLLM(apiKey string, opts ...Option) *AnthropicLLM {
	llm := &AnthropicLLM{
		APIKey:      apiKey,
		BaseURL:     defaultBaseURL,
		APIVersion:  defaultAPIVersion,
		Model:       defaultModel,
		MaxTokens:   defaultMaxTokens,
		Temperature: 0.7,
		TopP:        1.0,
		HttpClient:  &http.Client{},
	}

	// Apply options
	for _, opt := range opts {
		opt(llm)
	}

	return llm
}

// Option is a function that configures an AnthropicLLM.
type Option func(*AnthropicLLM)

// WithModel sets the model.
func WithModel(model string) Option {
	return func(a *AnthropicLLM) {
		a.Model = model
	}
}

// WithTemperature sets the temperature.
func WithTemperature(temperature float64) Option {
	return func(a *AnthropicLLM) {
		a.Temperature = temperature
	}
}

// WithMaxTokens sets the maximum number of tokens.
func WithMaxTokens(maxTokens int) Option {
	return func(a *AnthropicLLM) {
		a.MaxTokens = maxTokens
	}
}

// WithTopP sets the top_p value.
func WithTopP(topP float64) Option {
	return func(a *AnthropicLLM) {
		a.TopP = topP
	}
}

// WithTopK sets the top_k value.
func WithTopK(topK int) Option {
	return func(a *AnthropicLLM) {
		a.TopK = topK
	}
}

// WithBaseURL sets the base URL.
func WithBaseURL(baseURL string) Option {
	return func(a *AnthropicLLM) {
		a.BaseURL = baseURL
	}
}

// WithAPIVersion sets the Anthropic API version.
func WithAPIVersion(apiVersion string) Option {
	return func(a *AnthropicLLM) {
		a.APIVersion = apiVersion
	}
}

// WithHttpClient sets the HTTP client.
func WithHttpClient(client *http.Client) Option {
	return func(a *AnthropicLLM) {
		a.HttpClient = client
	}
}

// completionRequest represents a request to the Anthropic completion API.
type completionRequest struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens_to_sample"`
	Temperature float64 `json:"temperature,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
	StopSequences []string `json:"stop_sequences,omitempty"`
	Stream      bool    `json:"stream,omitempty"`
}

// completionResponse represents a response from the Anthropic completion API.
type completionResponse struct {
	Completion string `json:"completion"`
	StopReason string `json:"stop_reason"`
	Model      string `json:"model"`
}

// messageRequest represents a request to the Anthropic messages API.
type messageRequest struct {
	Model       string       `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int          `json:"max_tokens"`
	Temperature float64      `json:"temperature,omitempty"`
	TopP        float64      `json:"top_p,omitempty"`
	TopK        int          `json:"top_k,omitempty"`
	StopSequences []string   `json:"stop_sequences,omitempty"`
	Stream      bool         `json:"stream,omitempty"`
}

// chatMessage represents a message in the Anthropic API format.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// messageResponse represents a response from the Anthropic messages API.
type messageResponse struct {
	ID         string `json:"id"`
	Type       string `json:"type"`
	Role       string `json:"role"`
	Content    []messageContent `json:"content"`
	Model      string `json:"model"`
	StopReason string `json:"stop_reason"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// messageContent represents content in a message response.
type messageContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// Generate implements the BaseLanguageModel interface.
func (a *AnthropicLLM) Generate(ctx context.Context, prompts []llm.Prompt, options llm.ModelOptions) (llm.LLMResult, error) {
	result := llm.LLMResult{
		Generations: make([][]llm.Generation, len(prompts)),
		ModelInfo: map[string]interface{}{
			"model": a.Model,
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

		// Anthropic requires prompts to start with "\n\nHuman: " and end with "\n\nAssistant: "
		if promptStr != "" && !isAnthropicFormatted(promptStr) {
			promptStr = fmt.Sprintf("\n\nHuman: %s\n\nAssistant:", promptStr)
		}

		// Apply options
		temp := a.Temperature
		if options.Temperature != 0 {
			temp = options.Temperature
		}

		maxTokens := a.MaxTokens
		if options.MaxTokens != 0 {
			maxTokens = options.MaxTokens
		}

		topP := a.TopP
		if options.TopP != 0 {
			topP = options.TopP
		}

		model := a.Model
		if options.ModelName != "" {
			model = options.ModelName
		}

		// Create request
		reqBody := completionRequest{
			Model:       model,
			Prompt:      promptStr,
			MaxTokens:   maxTokens,
			Temperature: temp,
			TopP:        topP,
			TopK:        a.TopK,
			StopSequences: options.StopSequences,
		}

		// Serialize request
		jsonData, err := json.Marshal(reqBody)
		if err != nil {
			return result, fmt.Errorf("error marshaling request: %w", err)
		}

		// Create HTTP request
		req, err := http.NewRequestWithContext(ctx, "POST", a.BaseURL+defaultCompletionURL, bytes.NewBuffer(jsonData))
		if err != nil {
			return result, fmt.Errorf("error creating request: %w", err)
		}

		// Set headers
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-API-Key", a.APIKey)
		req.Header.Set("anthropic-version", a.APIVersion)

		// Make the request
		resp, err := a.HttpClient.Do(req)
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

		// Extract generation
		generation := llm.Generation{
			Text: compResp.Completion,
			Metadata: map[string]interface{}{
				"stop_reason": compResp.StopReason,
				"model":       compResp.Model,
			},
		}

		result.Generations[i] = []llm.Generation{generation}
	}

	return result, nil
}

// isAnthropicFormatted checks if a prompt is already formatted for Anthropic.
func isAnthropicFormatted(prompt string) bool {
	return len(prompt) >= 12 && prompt[:12] == "\n\nHuman: "
}

// GeneratePrompt implements the BaseLanguageModel interface.
func (a *AnthropicLLM) GeneratePrompt(ctx context.Context, prompt llm.Prompt, options llm.ModelOptions) ([]llm.Generation, error) {
	result, err := a.Generate(ctx, []llm.Prompt{prompt}, options)
	if err != nil {
		return nil, err
	}
	
	if len(result.Generations) == 0 || len(result.Generations[0]) == 0 {
		return nil, fmt.Errorf("no generations returned")
	}
	
	return result.Generations[0], nil
}

// Run implements the Runnable interface.
func (a *AnthropicLLM) Run(ctx context.Context, input llm.Prompt) ([]llm.Generation, error) {
	return a.GeneratePrompt(ctx, input, llm.ModelOptions{})
}

// RunWithConfig implements the Runnable interface.
func (a *AnthropicLLM) RunWithConfig(ctx context.Context, input llm.Prompt, config runnable.RunConfig) ([]llm.Generation, error) {
	return a.GeneratePrompt(ctx, input, llm.ModelOptions{
		Temperature: a.Temperature,
		MaxTokens:   a.MaxTokens,
		TopP:        a.TopP,
	})
}

// Stream implements the BaseLanguageModel interface.
func (a *AnthropicLLM) Stream(ctx context.Context, prompt llm.Prompt, options llm.ModelOptions) (<-chan llm.Generation, error) {
	// Implementation omitted for brevity - would use SSE for streaming
	return nil, fmt.Errorf("streaming not fully implemented")
}

// StreamWithConfig implements the Runnable interface.
func (a *AnthropicLLM) StreamWithConfig(ctx context.Context, input llm.Prompt, config runnable.RunConfig) (<-chan runnable.StreamingChunk[[]llm.Generation], error) {
	// Implementation omitted for brevity
	return nil, fmt.Errorf("streaming not fully implemented")
}

// GetInputSchema implements the Runnable interface.
func (a *AnthropicLLM) GetInputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "string",
		"description": "The prompt to generate completions for",
	}
}

// GetOutputSchema implements the Runnable interface.
func (a *AnthropicLLM) GetOutputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "array",
		"items": map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"text": map[string]interface{}{
					"type": "string",
					"description": "The generated completion text",
				},
			},
		},
	}
}

// ModelName returns the name of the model.
func (a *AnthropicLLM) ModelName() string {
	return a.Model
}

// ModelType returns the type of the model.
func (a *AnthropicLLM) ModelType() string {
	return "llm"
}

// AnthropicChat implements the ChatModel interface using Anthropic's API.
type AnthropicChat struct {
	*AnthropicLLM
}

// NewAnthropicChat creates a new AnthropicChat instance.
func NewAnthropicChat(apiKey string, opts ...Option) *AnthropicChat {
	return &AnthropicChat{
		AnthropicLLM: NewAnthropicLLM(apiKey, opts...),
	}
}

// GenerateChat implements the ChatModel interface.
func (a *AnthropicChat) GenerateChat(ctx context.Context, messages chat.MessageSet, options llm.ModelOptions) (chat.ChatResult, error) {
	// Convert LangChain messages to Anthropic format
	anthMessages := make([]chatMessage, len(messages))
	for i, msg := range messages {
		role := "user"
		if msg.Role == chat.RoleAssistant {
			role = "assistant"
		} else if msg.Role == chat.RoleSystem {
			// TODO: Handle system messages properly based on Anthropic API version
			// For now, prepend to first user message
			continue
		}

		// Extract content text
		content := ""
		for _, item := range msg.Content {
			if item.Type == chat.ContentTypeText {
				content += item.Text
			}
			// TODO: Handle other content types (images, etc.)
		}

		anthMessages[i] = chatMessage{
			Role:    role,
			Content: content,
		}
	}

	// Apply options
	temp := a.Temperature
	if options.Temperature != 0 {
		temp = options.Temperature
	}

	maxTokens := a.MaxTokens
	if options.MaxTokens != 0 {
		maxTokens = options.MaxTokens
	}

	topP := a.TopP
	if options.TopP != 0 {
		topP = options.TopP
	}

	model := a.Model
	if options.ModelName != "" {
		model = options.ModelName
	}

	// Create request
	reqBody := messageRequest{
		Model:       model,
		Messages:    anthMessages,
		MaxTokens:   maxTokens,
		Temperature: temp,
		TopP:        topP,
		TopK:        a.TopK,
		StopSequences: options.StopSequences,
	}

	// Serialize request
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return chat.ChatResult{}, fmt.Errorf("error marshaling request: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", a.BaseURL+defaultMessagesURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return chat.ChatResult{}, fmt.Errorf("error creating request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", a.APIKey)
	req.Header.Set("anthropic-version", a.APIVersion)

	// Make the request
	resp, err := a.HttpClient.Do(req)
	if err != nil {
		return chat.ChatResult{}, fmt.Errorf("error making request: %w", err)
	}
	defer resp.Body.Close()

	// Check for error response
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return chat.ChatResult{}, fmt.Errorf("API error (status %d): %s", resp.StatusCode, body)
	}

	// Parse response
	var msgResp messageResponse
	if err := json.NewDecoder(resp.Body).Decode(&msgResp); err != nil {
		return chat.ChatResult{}, fmt.Errorf("error decoding response: %w", err)
	}

	// Extract response content
	contentText := ""
	for _, content := range msgResp.Content {
		if content.Type == "text" {
			contentText += content.Text
		}
	}

	// Create result
	result := chat.ChatResult{
		Message: chat.Message{
			Role: chat.RoleAssistant,
			Content: []chat.ContentItem{
				{
					Type: chat.ContentTypeText,
					Text: contentText,
				},
			},
			CreatedAt: time.Now(),
			ID:        msgResp.ID,
			Metadata: map[string]interface{}{
				"stop_reason": msgResp.StopReason,
				"model":       msgResp.Model,
			},
		},
		Usage: llm.TokenUsage{
			PromptTokens:     msgResp.Usage.InputTokens,
			CompletionTokens: msgResp.Usage.OutputTokens,
			TotalTokens:      msgResp.Usage.InputTokens + msgResp.Usage.OutputTokens,
		},
		Metadata: map[string]interface{}{
			"model": msgResp.Model,
		},
	}

	return result, nil
}

// StreamChat implements the ChatModel interface.
func (a *AnthropicChat) StreamChat(ctx context.Context, messages chat.MessageSet, options llm.ModelOptions) (<-chan runnable.StreamingChunk[chat.ChatResult], error) {
	// Implementation omitted for brevity - would use SSE for streaming
	return nil, fmt.Errorf("streaming not fully implemented")
}

// SupportsToolCalls implements the ChatModel interface.
func (a *AnthropicChat) SupportsToolCalls() bool {
	// Check if the model supports tool calls
	// More recent Claude models (Claude 3, etc.) do support tool calls
	return a.Model == "claude-3-opus-20240229" || 
		a.Model == "claude-3-sonnet-20240229" || 
		a.Model == "claude-3-haiku-20240307"
}

// SupportsFunctionCalls implements the ChatModel interface.
func (a *AnthropicChat) SupportsFunctionCalls() bool {
	// Check if the model supports function calls
	// For now, treat this the same as tool calls
	return a.SupportsToolCalls()
}
