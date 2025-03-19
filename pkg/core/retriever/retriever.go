// pkg/core/retriever/retriever.go

package retriever

import (
	"context"

	"github.com/armyknife-tools/go-agent-framework/pkg/core/document"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/runnable"
)

// Retriever is the interface that all retrievers must implement.
type Retriever interface {
	// GetRelevantDocuments returns documents relevant to the query.
	GetRelevantDocuments(ctx context.Context, query string) ([]*document.Document, error)
}

// RunnableRetriever is a Runnable implementation of Retriever.
type RunnableRetriever interface {
	Retriever
	runnable.Runnable[string, []*document.Document]
}

// BaseRetriever provides a base implementation of the Runnable interface for retrievers.
type BaseRetriever struct {
	retriever Retriever
}

// NewBaseRetriever creates a new BaseRetriever with the given retriever.
func NewBaseRetriever(retriever Retriever) *BaseRetriever {
	return &BaseRetriever{
		retriever: retriever,
	}
}

// Run implements the Runnable interface.
func (br *BaseRetriever) Run(ctx context.Context, query string) ([]*document.Document, error) {
	return br.retriever.GetRelevantDocuments(ctx, query)
}

// RunWithConfig implements the Runnable interface.
func (br *BaseRetriever) RunWithConfig(ctx context.Context, query string, config runnable.RunConfig) ([]*document.Document, error) {
	return br.retriever.GetRelevantDocuments(ctx, query)
}

// Stream implements the Runnable interface.
func (br *BaseRetriever) Stream(ctx context.Context, query string) (<-chan runnable.StreamingChunk[[]*document.Document], error) {
	return br.StreamWithConfig(ctx, query, runnable.RunConfig{})
}

// StreamWithConfig implements the Runnable interface.
func (br *BaseRetriever) StreamWithConfig(ctx context.Context, query string, config runnable.RunConfig) (<-chan runnable.StreamingChunk[[]*document.Document], error) {
	outputCh := make(chan runnable.StreamingChunk[[]*document.Document], 1)

	go func() {
		defer close(outputCh)

		docs, err := br.retriever.GetRelevantDocuments(ctx, query)
		if err != nil {
			// Send error as metadata
			outputCh <- runnable.StreamingChunk[[]*document.Document]{
				Data:  nil,
				Index: 0,
				Final: true,
				Meta: map[string]interface{}{
					"error": err.Error(),
				},
			}
			return
		}

		outputCh <- runnable.StreamingChunk[[]*document.Document]{
			Data:  docs,
			Index: 0,
			Final: true,
		}
	}()

	return outputCh, nil
}

// GetInputSchema implements the Runnable interface.
func (br *BaseRetriever) GetInputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type":        "string",
		"description": "The query to retrieve documents for",
	}
}

// GetOutputSchema implements the Runnable interface.
func (br *BaseRetriever) GetOutputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "array",
		"items": map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"page_content": map[string]interface{}{
					"type":        "string",
					"description": "The content of the document",
				},
				"metadata": map[string]interface{}{
					"type":        "object",
					"description": "Metadata about the document",
				},
			},
		},
		"description": "The documents relevant to the query",
	}
}
