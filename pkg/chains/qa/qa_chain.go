// pkg/chains/qa/qa_chain.go

package qa

import (
	"context"
	"fmt"
	"strings"
	
	"github.com/armyknife-tools/go-agent-framework/pkg/core/document"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/llm"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/prompt"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/retriever"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/runnable"
)

// The prompt templates for the QA chain.
const (
	DefaultTemplate = `Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:`
)

// QAInput represents the input to the QA chain.
type QAInput struct {
	Question string                 `json:"question"`
	Context  string                 `json:"context,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// QAOutput represents the output from the QA chain.
type QAOutput struct {
	Answer      string                   `json:"answer"`
	SourceDocs  []*document.Document     `json:"source_docs,omitempty"`
	Metadata    map[string]interface{}   `json:"metadata,omitempty"`
}

// QAChainOptions contains options for the QA chain.
type QAChainOptions struct {
	Retriever  retriever.Retriever
	LLM        llm.LanguageModel
	Prompt     *prompt.PromptTemplate
	ReturnDocs bool
}

// QAChain is a chain for question answering.
type QAChain struct {
	retriever  retriever.Retriever
	llm        llm.LanguageModel
	prompt     *prompt.PromptTemplate
	returnDocs bool
}

// NewQAChain creates a new QAChain with the given options.
func NewQAChain(options QAChainOptions) (*QAChain, error) {
	if options.LLM == nil {
		return nil, fmt.Errorf("LLM is required")
	}
	
	// Create a default prompt if none is provided
	if options.Prompt == nil {
		var err error
		options.Prompt, err = prompt.NewPromptTemplate(DefaultTemplate, []string{"context", "question"})
		if err != nil {
			return nil, fmt.Errorf("error creating default prompt: %w", err)
		}
	}
	
	return &QAChain{
		retriever:  options.Retriever,
		llm:        options.LLM,
		prompt:     options.Prompt,
		returnDocs: options.ReturnDocs,
	}, nil
}

// Run executes the QA chain with the given input.
func (qac *QAChain) Run(ctx context.Context, input QAInput) (QAOutput, error) {
	// If context is provided, use it directly
	context := input.Context
	var documents []*document.Document
	
	// If no context is provided and a retriever is available, retrieve documents
	if context == "" && qac.retriever != nil {
		var err error
		documents, err = qac.retriever.GetRelevantDocuments(ctx, input.Question)
		if err != nil {
			return QAOutput{}, fmt.Errorf("error retrieving documents: %w", err)
		}
		
		// Format the retrieved documents into context
		var contextParts []string
		for i, doc := range documents {
			contextParts = append(contextParts, fmt.Sprintf("Document %d:\n%s", i+1, doc.PageContent))
		}
		context = strings.Join(contextParts, "\n\n")
	}
	
	// Format the prompt
	promptValues := map[string]interface{}{
		"context":  context,
		"question": input.Question,
	}
	
	formattedPrompt, err := qac.prompt.Format(promptValues)
	if err != nil {
		return QAOutput{}, fmt.Errorf("error formatting prompt: %w", err)
	}
	
	// Generate an answer using the LLM
	generations, err := qac.llm.GeneratePrompt(ctx, llm.StringPrompt(formattedPrompt), llm.ModelOptions{})
	if err != nil {
		return QAOutput{}, fmt.Errorf("error generating answer: %w", err)
	}
	
	if len(generations) == 0 {
		return QAOutput{}, fmt.Errorf("no answer generated")
	}
	
	// Create the output
	output := QAOutput{
		Answer: generations[0].Text,
		Metadata: map[string]interface{}{
			"prompt": formattedPrompt,
		},
	}
	
	// Include source documents if requested
	if qac.returnDocs {
		output.SourceDocs = documents
	}
	
	return output, nil
}

// RunWithConfig executes the QA chain with the given input and configuration.
func (qac *QAChain) RunWithConfig(ctx context.Context, input QAInput, config runnable.RunConfig) (QAOutput, error) {
	// For simplicity, we ignore the config
	return qac.Run(ctx, input)
}

// AsRunnable returns a runnable version of the QA chain.
func (qac *QAChain) AsRunnable() runnable.Runnable[QAInput, QAOutput] {
	return qac
}

// GetInputSchema implements the Runnable interface.
func (qac *QAChain) GetInputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"question": map[string]interface{}{
				"type":        "string",
				"description": "The question to answer",
			},
			"context": map[string]interface{}{
				"type":        "string",
				"description": "Optional context to use instead of retrieving documents",
			},
		},
		"required": []string{"question"},
	}
}

// GetOutputSchema implements the Runnable interface.
func (qac *QAChain) GetOutputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"answer": map[string]interface{}{
				"type":        "string",
				"description": "The generated answer",
			},
			"source_docs": map[string]interface{}{
				"type":  "array",
				"items": map[string]interface{}{"type": "object"},
				"description": "The source documents used to generate the answer",
			},
		},
	}
}

// Stream implements the Runnable interface.
func (qac *QAChain) Stream(ctx context.Context, input QAInput) (<-chan runnable.StreamingChunk[QAOutput], error) {
	return qac.StreamWithConfig(ctx, input, runnable.RunConfig{})
}

// StreamWithConfig implements the Runnable interface.
func (qac *QAChain) StreamWithConfig(ctx context.Context, input QAInput, config runnable.RunConfig) (<-chan runnable.StreamingChunk[QAOutput], error) {
	// Simple implementation that just runs the chain and returns the result as a single chunk
	outputCh := make(chan runnable.StreamingChunk[QAOutput], 1)
	
	go func() {
		defer close(outputCh)
		
		output, err := qac.RunWithConfig(ctx, input, config)
		if err != nil {
			outputCh <- runnable.StreamingChunk[QAOutput]{
				Data:  QAOutput{},
				Index: 0,
				Final: true,
				Meta: map[string]interface{}{
					"error": err.Error(),
				},
			}
			return
		}
		
		outputCh <- runnable.StreamingChunk[QAOutput]{
			Data:  output,
			Index: 0,
			Final: true,
		}
	}()
	
	return outputCh, nil
}
