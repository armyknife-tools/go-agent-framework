// pkg/core/chat/message.go

package chat

import (
	"context"
	"time"

	"github.com/armyknife-tools/go-agent-framework/pkg/core/llm"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/runnable"
)

// Role defines the role of a message sender.
type Role string

const (
	// RoleSystem represents a system message.
	RoleSystem Role = "system"
	// RoleUser represents a user message.
	RoleUser Role = "user"
	// RoleAssistant represents an assistant message.
	RoleAssistant Role = "assistant"
	// RoleFunction represents a function message.
	RoleFunction Role = "function"
	// RoleTool represents a tool message.
	RoleTool Role = "tool"
)

// ContentType defines the type of content in a message.
type ContentType string

const (
	// ContentTypeText represents plain text content.
	ContentTypeText ContentType = "text"
	// ContentTypeImage represents image content.
	ContentTypeImage ContentType = "image"
	// ContentTypeJSON represents JSON content.
	ContentTypeJSON ContentType = "json"
)

// ContentItem represents a single item of content in a message.
type ContentItem struct {
	Type ContentType             `json:"type"`
	Text string                  `json:"text,omitempty"`
	Data map[string]interface{}  `json:"data,omitempty"`
}

// Message represents a chat message.
type Message struct {
	Role          Role                   `json:"role"`
	Content       []ContentItem          `json:"content"`
	Name          string                 `json:"name,omitempty"`
	FunctionCall  *FunctionCall          `json:"function_call,omitempty"`
	ToolCalls     []ToolCall             `json:"tool_calls,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	ID            string                 `json:"id,omitempty"`
	CreatedAt     time.Time              `json:"created_at,omitempty"`
}

// FunctionCall represents a function call in a message.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolCall represents a tool call in a message.
type ToolCall struct {
	ID       string      `json:"id"`
	Type     string      `json:"type"`
	Function FunctionCall `json:"function"`
}

// NewTextMessage creates a new Message with text content.
func NewTextMessage(role Role, text string) Message {
	return Message{
		Role: role,
		Content: []ContentItem{
			{
				Type: ContentTypeText,
				Text: text,
			},
		},
		CreatedAt: time.Now(),
	}
}

// MessageSet represents a set of messages in a conversation.
type MessageSet []Message

// ChatResult represents the result of a chat model generation.
type ChatResult struct {
	Message  Message                `json:"message"`
	Usage    llm.TokenUsage         `json:"usage,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// ChatModel defines the interface for chat-based language models.
type ChatModel interface {
	// GenerateChat generates a chat response for the given messages.
	GenerateChat(ctx context.Context, messages MessageSet, options llm.ModelOptions) (ChatResult, error)
	
	// StreamChat generates a streaming chat response.
	StreamChat(ctx context.Context, messages MessageSet, options llm.ModelOptions) (<-chan runnable.StreamingChunk[ChatResult], error)
	
	// SupportsToolCalls returns whether the model supports tool calls.
	SupportsToolCalls() bool
	
	// SupportsFunctionCalls returns whether the model supports function calls.
	SupportsFunctionCalls() bool
	
	// ModelName returns the name of the model.
	ModelName() string
}

// RunnableChatModel is a Runnable implementation of ChatModel.
type RunnableChatModel interface {
	ChatModel
	runnable.Runnable[MessageSet, ChatResult]
}
