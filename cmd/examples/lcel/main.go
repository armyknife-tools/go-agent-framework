package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"
	
	"github.com/armyknife-tools/go-agent-framework/pkg/core/document"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/prompt"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/retriever"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/runnable"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/vectorstore"
	"github.com/armyknife-tools/go-agent-framework/pkg/llms/anthropic"
)

var (
	apiKey  = flag.String("api-key", "", "Anthropic API key")
	query   = flag.String("query", "", "Query to answer")
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
	
	fmt.Println("LangChain-Go LCEL Example")
	fmt.Println("=========================")
	
	// If query is not provided, prompt the user
	if *query == "" {
		fmt.Print("Enter your question: ")
		fmt.Scanln(query)
	}
	
	// Display the query
	fmt.Printf("\nQuestion: %s\n\n", *query)
	
	// Create a simple mock embedding model
	embedder := &MockEmbedder{}
	
	// Create a vector store with example documents
	store := vectorstore.NewMemoryVectorStore(embedder)
	err := store.AddDocuments(ctx, getExampleDocuments())
	if err != nil {
		log.Fatalf("Error adding documents to vector store: %v", err)
	}
	
	// Create a retriever
	vectorRetriever := retriever.NewVectorStoreRetriever(store, retriever.WithK(2))
	runnableRetriever := retriever.NewBaseRetriever(vectorRetriever)
	
	// Create a prompt template
	promptTemplate, err := prompt.NewRAGPrompt()
	if err != nil {
		log.Fatalf("Error creating prompt template: %v", err)
	}
	
	// Create a formatPrompt function using RunnableLambda
	formatPrompt := runnable.NewRunnableLambda(
		func(ctx context.Context, docs []*document.Document) (string, error) {
			// Format the documents into context
			var context string
			for i, doc := range docs {
				context += fmt.Sprintf("Document %d:\n%s\n\n", i+1, doc.PageContent)
			}
			return context, nil
		},
	)
	
	// Create a prompt formatter using RunnableMap
	promptFormatter := runnable.NewRunnableLambda(
		func(ctx context.Context, input struct {
			Question string
			Context  string
		}) (string, error) {
			// Format the prompt
			return promptTemplate.Format(map[string]interface{}{
				"question": input.Question,
				"context":  input.Context,
			})
		},
	)
	
	// Create a runnable chain using RunnableLambda for the query
	queryExtractor := runnable.NewRunnableLambda(
		func(ctx context.Context, input string) (struct{
			Question string
			Context  string
		}, error) {
			return struct{
				Question string
				Context  string
			}{
				Question: input,
			}, nil
		},
	)
	
	// Create a passthrough for parallel execution
	passthrough := runnable.NewRunnablePassthrough[string]()
	
	// Create our parallel runnables
	parallelRunnables := map[string]interface{}{
		"question": passthrough,
		"context": runnable.MustNewRunnableSequence[string, string](
			runnableRetriever,
			formatPrompt,
		),
	}
	
	// Create a parallel runner
	parallel, err := runnable.NewRunnableParallel[string, map[string]interface{}](parallelRunnables)
	if err != nil {
		log.Fatalf("Error creating parallel runner: %v", err)
	}
	
	// Create the final chain
	chain, err := runnable.NewRunnableSequence[string, string](
		queryExtractor,
		parallel,
		promptFormatter,
		llm,
	)
	if err != nil {
		log.Fatalf("Error creating chain: %v", err)
	}
	
	// Run the chain
	result, err := chain.Run(ctx, *query)
	if err != nil {
		log.Fatalf("Error running chain: %v", err)
	}
	
	// Print the result
	fmt.Printf("Answer: %s\n", result[0].Text)
}

// MockEmbedder is a simple embedding model for testing.
type MockEmbedder struct{}

// EmbedDocuments implements the EmbeddingModel interface.
func (m *MockEmbedder) EmbedDocuments(ctx context.Context, documents []string) ([][]float32, error) {
	embeddings := make([][]float32, len(documents))
	for i, doc := range documents {
		// Create a simple embedding - not realistic but for demonstration
		embeddings[i] = make([]float32, 10)
		for j := 0; j < 10 && j < len(doc); j++ {
			embeddings[i][j] = float32(doc[j] % 10)
		}
	}
	return embeddings, nil
}

// EmbedQuery implements the EmbeddingModel interface.
func (m *MockEmbedder) EmbedQuery(ctx context.Context, query string) ([]float32, error) {
	embedding := make([]float32, 10)
	for j := 0; j < 10 && j < len(query); j++ {
		embedding[j] = float32(query[j] % 10)
	}
	return embedding, nil
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
