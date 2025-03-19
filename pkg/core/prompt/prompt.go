// pkg/core/prompt/prompt.go

package prompt

import (
	"fmt"
	"regexp"
	"strings"
)

// PromptTemplate is a template for generating prompts.
type PromptTemplate struct {
	Template   string
	InputVariables []string
	
	// The regex to find variables in the template.
	variableRegex *regexp.Regexp
}

// NewPromptTemplate creates a new PromptTemplate with the given template and input variables.
func NewPromptTemplate(template string, inputVariables []string) (*PromptTemplate, error) {
	pt := &PromptTemplate{
		Template:       template,
		InputVariables: inputVariables,
		variableRegex:  regexp.MustCompile(`{([^{}]+)}`),
	}
	
	// Validate the template
	if err := pt.validate(); err != nil {
		return nil, err
	}
	
	return pt, nil
}

// validate checks that the template contains all the input variables.
func (pt *PromptTemplate) validate() error {
	// Find all variables in the template
	matches := pt.variableRegex.FindAllStringSubmatch(pt.Template, -1)
	
	// Create a map of variables in the template
	templateVars := make(map[string]bool)
	for _, match := range matches {
		if len(match) > 1 {
			templateVars[match[1]] = true
		}
	}
	
	// Check that all input variables are in the template
	for _, v := range pt.InputVariables {
		if !templateVars[v] {
			return fmt.Errorf("input variable %q not found in template", v)
		}
	}
	
	// Check that all template variables are in the input variables
	for v := range templateVars {
		found := false
		for _, iv := range pt.InputVariables {
			if v == iv {
				found = true
				break
			}
		}
		
		if !found {
			return fmt.Errorf("template variable %q not found in input variables", v)
		}
	}
	
	return nil
}

// Format formats the template with the given values.
func (pt *PromptTemplate) Format(values map[string]interface{}) (string, error) {
	// Create a copy of the template
	result := pt.Template
	
	// Replace variables with their values
	for _, v := range pt.InputVariables {
		value, ok := values[v]
		if !ok {
			return "", fmt.Errorf("missing value for input variable %q", v)
		}
		
		// Convert value to string
		var strValue string
		switch val := value.(type) {
		case string:
			strValue = val
		case fmt.Stringer:
			strValue = val.String()
		default:
			strValue = fmt.Sprintf("%v", val)
		}
		
		// Replace the variable in the template
		placeholder := fmt.Sprintf("{%s}", v)
		result = strings.ReplaceAll(result, placeholder, strValue)
	}
	
	return result, nil
}

// FormatPrompt formats the template with the given values and returns a StringPrompt.
func (pt *PromptTemplate) FormatPrompt(values map[string]interface{}) (string, error) {
	return pt.Format(values)
}

// ChatPromptTemplate is a template for generating chat prompts.
type ChatPromptTemplate struct {
	Messages []MessageTemplate
}

// MessageTemplate is a template for a single message in a chat.
type MessageTemplate struct {
	Role            string
	ContentTemplate *PromptTemplate
}

// NewChatPromptTemplate creates a new ChatPromptTemplate with the given message templates.
func NewChatPromptTemplate(messages []MessageTemplate) *ChatPromptTemplate {
	return &ChatPromptTemplate{
		Messages: messages,
	}
}

// FormatMessages formats the chat prompt templates with the given values.
func (cpt *ChatPromptTemplate) FormatMessages(values map[string]interface{}) ([]map[string]string, error) {
	var formattedMessages []map[string]string
	
	for _, msg := range cpt.Messages {
		content, err := msg.ContentTemplate.Format(values)
		if err != nil {
			return nil, fmt.Errorf("error formatting message content: %w", err)
		}
		
		formattedMessages = append(formattedMessages, map[string]string{
			"role":    msg.Role,
			"content": content,
		})
	}
	
	return formattedMessages, nil
}

// Common prompt templates.
var (
	// DefaultQuestionAnsweringPrompt is a default prompt for question answering.
	DefaultQuestionAnsweringPrompt = `Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:`
	
	// DefaultRAGPrompt is a default prompt for retrieval-augmented generation.
	DefaultRAGPrompt = `You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

{context}

Question: {question}
Answer:`
	
	// DefaultSummarizationPrompt is a default prompt for summarization.
	DefaultSummarizationPrompt = `Write a concise summary of the following text:

{text}

CONCISE SUMMARY:`
)

// Common convenience functions to create prompt templates.
var (
	// NewQuestionAnsweringPrompt creates a new prompt template for question answering.
	NewQuestionAnsweringPrompt = func() (*PromptTemplate, error) {
		return NewPromptTemplate(DefaultQuestionAnsweringPrompt, []string{"context", "question"})
	}
	
	// NewRAGPrompt creates a new prompt template for retrieval-augmented generation.
	NewRAGPrompt = func() (*PromptTemplate, error) {
		return NewPromptTemplate(DefaultRAGPrompt, []string{"context", "question"})
	}
	
	// NewSummarizationPrompt creates a new prompt template for summarization.
	NewSummarizationPrompt = func() (*PromptTemplate, error) {
		return NewPromptTemplate(DefaultSummarizationPrompt, []string{"text"})
	}
)
