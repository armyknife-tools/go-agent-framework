// pkg/core/vectorstore/vectorstore.go

package vectorstore

import (
	"context"

	"github.com/armyknife-tools/go-agent-framework/pkg/core/document"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/embedding"
)

// FilterFunc is a function that filters documents based on metadata.
type FilterFunc func(map[string]interface{}) bool

// DocumentScore represents a document with a similarity score.
type DocumentScore struct {
	Document *document.Document
	Score    float32
}

// VectorStore is the interface that all vector stores must implement.
type VectorStore interface {
	// AddDocuments adds documents to the vector store.
	AddDocuments(ctx context.Context, docs []*document.Document) error

	// AddDocumentsWithEmbeddings adds documents with pre-computed embeddings.
	AddDocumentsWithEmbeddings(ctx context.Context, docs []*document.Document, embeddings [][]float32) error

	// SimilaritySearch searches for documents similar to the query.
	SimilaritySearch(ctx context.Context, query string, k int, filter FilterFunc) ([]*document.Document, error)

	// SimilaritySearchWithScore searches for documents and returns similarity scores.
	SimilaritySearchWithScore(ctx context.Context, query string, k int, filter FilterFunc) ([]DocumentScore, error)

	// Delete deletes documents from the vector store.
	Delete(ctx context.Context, ids []string) error

	// GetEmbedder returns the embedding model used by the vector store.
	GetEmbedder() embedding.EmbeddingModel
}
