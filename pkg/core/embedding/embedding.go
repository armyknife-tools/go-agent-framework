// pkg/core/embedding/embedding.go

package embedding

import (
	"context"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	
	"github.com/armyknife-tools/go-agent-framework/pkg/core/runnable"
)

// EmbeddingModel defines the interface for embedding models.
type EmbeddingModel interface {
	// EmbedDocuments embeds a slice of documents.
	EmbedDocuments(ctx context.Context, documents []string) ([][]float32, error)
	
	// EmbedQuery embeds a single query.
	EmbedQuery(ctx context.Context, query string) ([]float32, error)
	
	// BatchSize returns the recommended batch size for this model.
	BatchSize() int
	
	// ModelName returns the name of the model.
	ModelName() string
	
	// Dimension returns the embedding vector dimension.
	Dimension() int
}

// RunnableEmbedding is a Runnable implementation of EmbeddingModel.
type RunnableEmbedding interface {
	EmbeddingModel
	runnable.Runnable[string, []float32]
}

// OpenAIEmbedding implements an embedding model using OpenAI's API.
type OpenAIEmbedding struct {
	APIKey       string
	BaseURL      string
	Model        string
	batchSize    int
	dimension    int
	HttpClient   *http.Client
	Organization string
}

// OpenAIEmbeddingOption is a function that configures an OpenAIEmbedding.
type OpenAIEmbeddingOption func(*OpenAIEmbedding)

const (
	defaultEmbeddingBaseURL   = "https://api.openai.com/v1"
	defaultEmbeddingModel     = "text-embedding-ada-002"
	defaultEmbeddingDimension = 1536
	defaultEmbeddingBatchSize = 100
)

// WithEmbeddingModel sets the model.
func WithEmbeddingModel(model string) OpenAIEmbeddingOption {
	return func(o *OpenAIEmbedding) {
		o.Model = model
	}
}

// WithEmbeddingBaseURL sets the base URL.
func WithEmbeddingBaseURL(baseURL string) OpenAIEmbeddingOption {
	return func(o *OpenAIEmbedding) {
		o.BaseURL = baseURL
	}
}

// WithEmbeddingBatchSize sets the batch size.
func WithEmbeddingBatchSize(batchSize int) OpenAIEmbeddingOption {
	return func(o *OpenAIEmbedding) {
		o.batchSize = batchSize
	}
}

// WithEmbeddingDimension sets the embedding dimension.
func WithEmbeddingDimension(dimension int) OpenAIEmbeddingOption {
	return func(o *OpenAIEmbedding) {
		o.dimension = dimension
	}
}

// WithEmbeddingHttpClient sets the HTTP client.
func WithEmbeddingHttpClient(client *http.Client) OpenAIEmbeddingOption {
	return func(o *OpenAIEmbedding) {
		o.HttpClient = client
	}
}

// WithEmbeddingOrganization sets the OpenAI organization.
func WithEmbeddingOrganization(org string) OpenAIEmbeddingOption {
	return func(o *OpenAIEmbedding) {
		o.Organization = org
	}
}

// NewOpenAIEmbedding creates a new OpenAIEmbedding.
func NewOpenAIEmbedding(apiKey string, opts ...OpenAIEmbeddingOption) *OpenAIEmbedding {
	e := &OpenAIEmbedding{
		APIKey:     apiKey,
		BaseURL:    defaultEmbeddingBaseURL,
		Model:      defaultEmbeddingModel,
		batchSize:  defaultEmbeddingBatchSize,
		dimension:  defaultEmbeddingDimension,
		HttpClient: &http.Client{},
	}

	// Apply options
	for _, opt := range opts {
		opt(e)
	}

	return e
}

// embeddingRequest represents a request to the OpenAI embeddings API.
type embeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// embeddingResponse represents a response from the OpenAI embeddings API.
type embeddingResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// EmbedDocuments embeds a slice of documents.
func (o *OpenAIEmbedding) EmbedDocuments(ctx context.Context, documents []string) ([][]float32, error) {
	// Process in batches
	var allEmbeddings [][]float32
	for i := 0; i < len(documents); i += o.batchSize {
		end := i + o.batchSize
		if end > len(documents) {
			end = len(documents)
		}
		
		batch := documents[i:end]
		embeddings, err := o.embedBatch(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("error embedding batch: %w", err)
		}
		
		allEmbeddings = append(allEmbeddings, embeddings...)
	}
	
	return allEmbeddings, nil
}

// embedBatch embeds a batch of documents.
func (o *OpenAIEmbedding) embedBatch(ctx context.Context, batch []string) ([][]float32, error) {
	// Create request
	reqBody := embeddingRequest{
		Model: o.Model,
		Input: batch,
	}
	
	// Serialize request
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %w", err)
	}
	
	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s/embeddings", o.BaseURL), bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
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
		return nil, fmt.Errorf("error making request: %w", err)
	}
	defer resp.Body.Close()
	
	// Check for error response
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, body)
	}
	
	// Parse response
	var embResp embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&embResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}
	
	// Extract embeddings
	embeddings := make([][]float32, len(embResp.Data))
	for _, item := range embResp.Data {
		embeddings[item.Index] = item.Embedding
	}
	
	return embeddings, nil
}

// EmbedQuery embeds a single query.
func (o *OpenAIEmbedding) EmbedQuery(ctx context.Context, query string) ([]float32, error) {
	embeddings, err := o.EmbedDocuments(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}
	
	return embeddings[0], nil
}

// BatchSize returns the recommended batch size for this model.
func (o *OpenAIEmbedding) BatchSize() int {
	return o.batchSize
}

// ModelName returns the name of the model.
func (o *OpenAIEmbedding) ModelName() string {
	return o.Model
}

// Dimension returns the embedding vector dimension.
func (o *OpenAIEmbedding) Dimension() int {
	return o.dimension
}

// Run implements the Runnable interface.
func (o *OpenAIEmbedding) Run(ctx context.Context, input string) ([]float32, error) {
	return o.EmbedQuery(ctx, input)
}

// RunWithConfig implements the Runnable interface.
func (o *OpenAIEmbedding) RunWithConfig(ctx context.Context, input string, config runnable.RunConfig) ([]float32, error) {
	return o.EmbedQuery(ctx, input)
}

// Stream is not applicable for embeddings and returns an error.
func (o *OpenAIEmbedding) Stream(ctx context.Context, input string) (<-chan runnable.StreamingChunk[[]float32], error) {
	return nil, fmt.Errorf("streaming not supported for embeddings")
}

// StreamWithConfig is not applicable for embeddings and returns an error.
func (o *OpenAIEmbedding) StreamWithConfig(ctx context.Context, input string, config runnable.RunConfig) (<-chan runnable.StreamingChunk[[]float32], error) {
	return nil, fmt.Errorf("streaming not supported for embeddings")
}

// GetInputSchema implements the Runnable interface.
func (o *OpenAIEmbedding) GetInputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "string",
		"description": "The text to embed",
	}
}

// GetOutputSchema implements the Runnable interface.
func (o *OpenAIEmbedding) GetOutputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "array",
		"items": map[string]interface{}{
			"type": "number",
			"format": "float",
		},
		"description": "The embedding vector",
	}
}