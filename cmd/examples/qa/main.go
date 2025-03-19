package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"
	
	"github.com/armyknife-tools/go-agent-framework/pkg/chains/qa"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/document"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/embedding"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/prompt"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/retriever"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/vectorstore"
	"github.com/armyknife-tools/go-agent-framework/pkg/llms/anthropic"
)

var (
	apiKey  = flag.String("api-key", "", "Anthropic API key")
	query   = flag.String("query", "", "Query to answer")
	docFile = flag.String("doc", "", "Document file to use")
)

func main() {
	flag.Parse()
	
	// Check if API key is provided
	if *apiKey == "" {
		// Try to get API key from environment
		*apiKey = os.Getenv("ANTHROPIC_API_KEY")
		if *apiKey == "" {
			log.Fatal("API key not provided. Use --api-key flag or ANTHROPIC_API_KEY environment variable.")
		}
	}
	
	// Create a context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	// Create the LLM
	llm := anthropic.NewAnthropicLLM(*apiKey, 
		anthropic.WithModel("claude-3-opus-20240229"),
		anthropic.WithTemperature(0.0),
		anthropic.WithMaxTokens(2000),
	)
	
	fmt.Println("LangChain-Go Question Answering Example")
	fmt.Println("=======================================")
	
	// If query is not provided, prompt the user
	if *query == "" {
		fmt.Print("Enter your question: ")
		fmt.Scanln(query)
	}
	
	// Load the document if provided
	var docs []*document.Document
	if *docFile != "" {
		content, err := os.ReadFile(*docFile)
		if err != nil {
			log.Fatalf("Error reading document file: %v", err)
		}
		
		// Create a simple document
		docs = append(docs, document.NewDocument(string(content), map[string]interface{}{
			"source": *docFile,
		}))
		
		fmt.Printf("Loaded document from %s\n", *docFile)
	} else {
		// Use example documents
		docs = getExampleDocuments()
		fmt.Println("Using example documents")
	}
	
	// Display the query
	fmt.Printf("\nQuestion: %s\n\n", *query)
	
	// Use a simple mock embedding model for this example
	// In a real application, you would use a real embedding model
	embedder := &MockEmbedder{}
	
	// Create a vector store and add the documents
	store := vectorstore.NewMemoryVectorStore(embedder)
	err := store.AddDocuments(ctx, docs)
	if err != nil {
		log.Fatalf("Error adding documents to vector store: %v", err)
	}
	
	// Create a retriever
	vectorRetriever := retriever.NewVectorStoreRetriever(store, retriever.WithK(2))
	
	// Create a QA prompt template
	qaPrompt, err := prompt.NewRAGPrompt()
	if err != nil {
		log.Fatalf("Error creating QA prompt: %v", err)
	}
	
	// Create a QA chain
	qaChain, err := qa.NewQAChain(qa.QAChainOptions{
		Retriever:  vectorRetriever,
		LLM:        llm,
		Prompt:     qaPrompt,
		ReturnDocs: true,
	})
	if err != nil {
		log.Fatalf("Error creating QA chain: %v", err)
	}
	
	// Run the QA chain
	result, err := qaChain.Run(ctx, qa.QAInput{
		Question: *query,
	})
	if err != nil {
		log.Fatalf("Error running QA chain: %v", err)
	}
	
	// Print the answer
	fmt.Printf("Answer: %s\n\n", result.Answer)
	
	// Print source documents
	if len(result.SourceDocs) > 0 {
		fmt.Println("Source Documents:")
		for i, doc := range result.SourceDocs {
			fmt.Printf("Document %d:", i+1)
			if source, ok := doc.Metadata["source"]; ok {
				fmt.Printf(" (Source: %v)", source)
			}
			fmt.Println()
			
			// Print a snippet of the document
			const maxLength = 200
			content := doc.PageContent
			if len(content) > maxLength {
				content = content[:maxLength] + "..."
			}
			fmt.Printf("  %s\n\n", content)
		}
	}
}

// MockEmbedder is a simple embedding model for testing.
type MockEmbedder struct{}

// EmbedDocuments implements the EmbeddingModel interface.
func (m *MockEmbedder) EmbedDocuments(ctx context.Context, documents []string) ([][]float32, error) {
	embeddings := make([][]float32, len(documents))
	for i, doc := range documents {
		embeddings[i] = m.getSimpleEmbedding(doc)
	}
	return embeddings, nil
}

// EmbedQuery implements the EmbeddingModel interface.
func (m *MockEmbedder) EmbedQuery(ctx context.Context, query string) ([]float32, error) {
	return m.getSimpleEmbedding(query), nil
}

// getSimpleEmbedding creates a simple embedding based on the text.
// This is a very naive implementation for demonstration purposes only.
func (m *MockEmbedder) getSimpleEmbedding(text string) []float32 {
	// In a real application, you would use a real embedding model
	// This implementation just creates a simple embedding based on word presence
	
	// Use a fixed dimension
	dim := 10
	embedding := make([]float32, dim)
	
	// Some simple "features" to track
	keywords := []string{"go", "language", "chain", "model", "embedding", "vector", "retriever", "question", "answer", "document"}
	
	for i, keyword := range keywords {
		// Set the i-th dimension based on the presence of the keyword
		if strings.Contains(strings.ToLower(text), keyword) {
			embedding[i] = 1.0
		}
	}
	
	return embedding
}

// BatchSize implements the EmbeddingModel interface.
func (m *MockEmbedder) BatchSize() int {
	return 10
}

// ModelName implements the EmbeddingModel interface.
func (m *MockEmbedder) ModelName() string {
	return "mock-embedder"
}

// Dimension implements the EmbeddingModel interface.
func (m *MockEmbedder) Dimension() int {
	return 10
}

// getExampleDocuments returns some example documents for testing.
func getExampleDocuments() []*document.Document {
	docs := []*document.Document{
		document.NewDocument(
			"Go is a statically typed, compiled programming language designed at Google. "+
				"Go is syntactically similar to C, but with memory safety, garbage collection, "+
				"structural typing, and CSP-style concurrency.",
			map[string]interface{}{
				"source": "go-lang-info.txt",
				"topic":  "programming",
			},
		),
		document.NewDocument(
			"The LangChain framework is designed to simplify the creation of applications "+
				"using large language models (LLMs). It provides a standard interface for chains, "+
				"lots of integrations with other tools, and end-to-end chains for common applications.",
			map[string]interface{}{
				"source": "langchain-info.txt",
				"topic":  "ai",
			},
		),
		document.NewDocument(
			"Embeddings in natural language processing are vector representations of text. "+
				"These vectors capture semantic meaning, allowing machines to understand relationships "+
				"between words and concepts. They are fundamental to many NLP tasks.",
			map[string]interface{}{
				"source": "embeddings-info.txt",
				"topic":  "ai",
			},
		),
		document.NewDocument(
			"Vector stores are specialized databases designed to store and search vector embeddings. "+
				"They enable semantic search by finding vectors that are similar to a query vector. "+
				"Popular vector stores include Pinecone, Weaviate, and Milvus.",
			map[string]interface{}{
				"source": "vectorstores-info.txt",
				"topic":  "databases",
			},
		),
	}
	return docs
}