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

// memory_vectorstore.go

package vectorstore

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	
	"github.com/armyknife-tools/go-agent-framework/pkg/core/document"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/embedding"
)

// MemoryVectorStore is an in-memory implementation of VectorStore.
type MemoryVectorStore struct {
	embedder   embedding.EmbeddingModel
	documents  map[string]*document.Document
	embeddings map[string][]float32
	mu         sync.RWMutex
}

// NewMemoryVectorStore creates a new MemoryVectorStore with the given embedder.
func NewMemoryVectorStore(embedder embedding.EmbeddingModel) *MemoryVectorStore {
	return &MemoryVectorStore{
		embedder:   embedder,
		documents:  make(map[string]*document.Document),
		embeddings: make(map[string][]float32),
	}
}

// AddDocuments adds documents to the vector store.
func (m *MemoryVectorStore) AddDocuments(ctx context.Context, docs []*document.Document) error {
	if len(docs) == 0 {
		return nil
	}
	
	// Extract the contents
	texts := make([]string, len(docs))
	for i, doc := range docs {
		texts[i] = doc.PageContent
	}
	
	// Generate embeddings
	embeddings, err := m.embedder.EmbedDocuments(ctx, texts)
	if err != nil {
		return fmt.Errorf("error generating embeddings: %w", err)
	}
	
	return m.AddDocumentsWithEmbeddings(ctx, docs, embeddings)
}

// AddDocumentsWithEmbeddings adds documents with pre-computed embeddings.
func (m *MemoryVectorStore) AddDocumentsWithEmbeddings(ctx context.Context, docs []*document.Document, embeddings [][]float32) error {
	if len(docs) != len(embeddings) {
		return fmt.Errorf("number of documents (%d) does not match number of embeddings (%d)", len(docs), len(embeddings))
	}
	
	m.mu.Lock()
	defer m.mu.Unlock()
	
	for i, doc := range docs {
		id := doc.GetID()
		m.documents[id] = doc
		m.embeddings[id] = embeddings[i]
	}
	
	return nil
}

// SimilaritySearch searches for documents similar to the query.
func (m *MemoryVectorStore) SimilaritySearch(ctx context.Context, query string, k int, filter FilterFunc) ([]*document.Document, error) {
	results, err := m.SimilaritySearchWithScore(ctx, query, k, filter)
	if err != nil {
		return nil, err
	}
	
	docs := make([]*document.Document, len(results))
	for i, result := range results {
		docs[i] = result.Document
	}
	
	return docs, nil
}

// SimilaritySearchWithScore searches for documents and returns similarity scores.
func (m *MemoryVectorStore) SimilaritySearchWithScore(ctx context.Context, query string, k int, filter FilterFunc) ([]DocumentScore, error) {
	// Generate embedding for the query
	queryEmbedding, err := m.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("error embedding query: %w", err)
	}
	
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	// Calculate similarities
	var results []DocumentScore
	for id, docEmbedding := range m.embeddings {
		doc := m.documents[id]
		
		// Apply filter if provided
		if filter != nil && !filter(doc.Metadata) {
			continue
		}
		
		// Calculate cosine similarity
		score := cosineSimilarity(queryEmbedding, docEmbedding)
		
		results = append(results, DocumentScore{
			Document: doc,
			Score:    score,
		})
	}
	
	// Sort by score (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	
	// Limit to k results
	if k > 0 && len(results) > k {
		results = results[:k]
	}
	
	return results, nil
}

// Delete deletes documents from the vector store.
func (m *MemoryVectorStore) Delete(ctx context.Context, ids []string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	for _, id := range ids {
		delete(m.documents, id)
		delete(m.embeddings, id)
	}
	
	return nil
}

// GetEmbedder returns the embedding model used by the vector store.
func (m *MemoryVectorStore) GetEmbedder() embedding.EmbeddingModel {
	return m.embedder
}

// cosineSimilarity calculates the cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	
	var dotProduct float32
	var normA float32
	var normB float32
	
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	
	if normA == 0 || normB == 0 {
		return 0
	}
	
	return float32(dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB)))))
}