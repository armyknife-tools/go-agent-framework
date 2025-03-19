# Go Agent Framework

A powerful Go framework for building LLM-powered applications and agents.

## Overview

Go Agent Framework is a comprehensive toolkit designed to simplify the development of LLM-powered applications in Go. Inspired by frameworks like LangChain, it provides a modular and extensible architecture for building complex AI applications, including question-answering systems, document processing, and conversational agents.

## Features

- **Modular Architecture**: Easily compose complex chains and workflows using reusable components
- **LLM Integration**: Ready-to-use integrations with popular LLM providers (OpenAI, Anthropic)
- **Retrieval-Augmented Generation (RAG)**: Built-in support for document retrieval and context-aware generation
- **Streaming Support**: Stream responses from LLMs for better user experience
- **Extensible Design**: Implement custom components that fit seamlessly into the framework
- **Type-Safe Interfaces**: Leverage Go's type system for safer code

## Installation

```bash
go get github.com/armyknife-tools/go-agent-framework
```

## Quick Start

Here's a simple example of using the framework to create a question-answering system:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/armyknife-tools/go-agent-framework/pkg/chains/qa"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/document"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/prompt"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/retriever"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/vectorstore"
	"github.com/armyknife-tools/go-agent-framework/pkg/llms/anthropic"
)

func main() {
	// Get API key from environment
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		log.Fatal("ANTHROPIC_API_KEY environment variable is required")
	}

	// Create a context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Create the LLM
	llm := anthropic.NewAnthropicLLM(apiKey,
		anthropic.WithModel("claude-3-sonnet-20250219"),
		anthropic.WithTemperature(0.0),
	)

	// Create documents
	docs := []*document.Document{
		document.NewDocument("Go is a statically typed, compiled programming language.", nil),
		document.NewDocument("Go was designed at Google by Robert Griesemer, Rob Pike, and Ken Thompson.", nil),
	}

	// Create a simple embedding model
	embedder := &MockEmbedder{} // Implement your own or use a real embedder

	// Create a vector store and add documents
	store := vectorstore.NewMemoryVectorStore(embedder)
	if err := store.AddDocuments(ctx, docs); err != nil {
		log.Fatalf("Error adding documents: %v", err)
	}

	// Create a retriever
	vectorRetriever := retriever.NewVectorStoreRetriever(store)

	// Create a QA prompt template
	qaPrompt, err := prompt.NewRAGPrompt()
	if err != nil {
		log.Fatalf("Error creating prompt: %v", err)
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
		Question: "Who designed the Go programming language?",
	})
	if err != nil {
		log.Fatalf("Error running QA chain: %v", err)
	}

	// Print the answer
	fmt.Printf("Answer: %s\n", result.Answer)
}

```

## Core Components

### Language Models (LLMs)

The framework provides a unified interface for working with various language models:

```go
// Create an Anthropic LLM
llm := anthropic.NewAnthropicLLM(apiKey,
    anthropic.WithModel("claude-3-sonnet-20250219"),
    anthropic.WithTemperature(0.0),
)

// Create an OpenAI LLM
llm := openai.NewOpenAILLM(apiKey,
    openai.WithModel("gpt-4"),
    openai.WithTemperature(0.7),
)
```

### Prompts

Create and format prompts with dynamic variables:

```go
// Create a prompt template
template := "Answer the question based on the context.\nContext: {context}\nQuestion: {question}\nAnswer:"
promptTemplate, err := prompt.NewPromptTemplate(template, []string{"context", "question"})

// Format the prompt
formattedPrompt, err := promptTemplate.Format(map[string]interface{}{
    "context":  "Go was created at Google in 2007.",
    "question": "When was Go created?",
})
```

### Retrieval

Retrieve relevant documents from a vector store:

```go
// Create a vector store
store := vectorstore.NewMemoryVectorStore(embedder)
store.AddDocuments(ctx, documents)

// Create a retriever
retriever := retriever.NewVectorStoreRetriever(store, retriever.WithK(3))

// Retrieve documents
docs, err := retriever.GetRelevantDocuments(ctx, "When was Go created?")
```

### Chains

Combine components into reusable chains:

```go
// Create a QA chain
qaChain, err := qa.NewQAChain(qa.QAChainOptions{
    Retriever:  retriever,
    LLM:        llm,
    Prompt:     promptTemplate,
    ReturnDocs: true,
})

// Run the chain
result, err := qaChain.Run(ctx, qa.QAInput{
    Question: "When was Go created?",
})
```

### LangChain Expression Language (LCEL)

Build complex chains using a composable API:

```go
// Create components
retriever := retriever.NewBaseRetriever(vectorRetriever)
promptTemplate, _ := prompt.NewRAGPrompt()

// Create a runnable chain
chain, err := runnable.NewRunnableSequence[string, string](
    queryExtractor,
    parallel,
    promptFormatter,
    llm,
)

// Run the chain
result, err := chain.Run(ctx, "When was Go created?")
```

## Architecture

The framework is organized into the following packages:

- **pkg/core**: Core interfaces and implementations
  - **llm**: Language model interfaces and base implementations
  - **prompt**: Prompt templates and formatting
  - **document**: Document representation and handling
  - **retriever**: Document retrieval interfaces and implementations
  - **vectorstore**: Vector storage for embeddings
  - **runnable**: Composable runnable components
  - **embedding**: Embedding models and utilities
  - **chat**: Chat message types and utilities

- **pkg/llms**: LLM provider implementations
  - **openai**: OpenAI API integration
  - **anthropic**: Anthropic API integration

- **pkg/chains**: Pre-built chains for common tasks
  - **qa**: Question-answering chains

## Design Considerations

### Interface Design

The framework uses Go's interface system to provide a flexible and extensible architecture. One notable design challenge is with the `LanguageModel` interface, which attempts to embed both `BaseLanguageModel` and `runnable.Runnable[Prompt, []Generation]`. This creates a method conflict with the `Stream` method that exists in both interfaces.

To work around this issue, we recommend:

1. Using the `ToRunnable` function to convert a `LanguageModel` to a `Runnable` when needed
2. Implementing the interfaces separately and using composition rather than embedding
3. Using the `RunnableLanguageModel` adapter that properly implements both interfaces

### Streaming Support

The framework provides streaming support for LLMs through the `Stream` method:

```go
// Stream responses from an LLM
genChan, err := llm.Stream(ctx, prompt, options)
for gen := range genChan {
    fmt.Print(gen.Text)
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the LICENSE file included in the repository.
