// pkg/core/retriever/vectorstore_retriever.go

package retriever

import (
	"context"

	"github.com/armyknife-tools/go-agent-framework/pkg/core/document"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/vectorstore"
)

// VectorStoreRetriever retrieves documents from a vector store.
type VectorStoreRetriever struct {
	vectorStore vectorstore.VectorStore
	k           int
	filter      vectorstore.FilterFunc
}

// VectorStoreRetrieverOption is a function that configures a VectorStoreRetriever.
type VectorStoreRetrieverOption func(*VectorStoreRetriever)

// WithK sets the number of documents to retrieve.
func WithK(k int) VectorStoreRetrieverOption {
	return func(r *VectorStoreRetriever) {
		r.k = k
	}
}

// WithFilter sets the filter function.
func WithFilter(filter vectorstore.FilterFunc) VectorStoreRetrieverOption {
	return func(r *VectorStoreRetriever) {
		r.filter = filter
	}
}

// NewVectorStoreRetriever creates a new VectorStoreRetriever.
func NewVectorStoreRetriever(vectorStore vectorstore.VectorStore, opts ...VectorStoreRetrieverOption) *VectorStoreRetriever {
	r := &VectorStoreRetriever{
		vectorStore: vectorStore,
		k:           4, // Default to 4 documents
	}

	for _, opt := range opts {
		opt(r)
	}

	return r
}

// GetRelevantDocuments implements the Retriever interface.
func (r *VectorStoreRetriever) GetRelevantDocuments(ctx context.Context, query string) ([]*document.Document, error) {
	return r.vectorStore.SimilaritySearch(ctx, query, r.k, r.filter)
}

// Runnable returns a Runnable version of the retriever.
func (r *VectorStoreRetriever) Runnable() RunnableRetriever {
	return NewBaseRetriever(r)
}
