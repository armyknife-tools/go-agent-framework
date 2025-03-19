// pkg/core/document/document.go

package document

import (
	"time"
)

// Document represents a document in the system.
type Document struct {
	// PageContent is the main content of the document.
	PageContent string `json:"page_content"`
	
	// Metadata is a map of metadata about the document.
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	
	// ID is an optional unique identifier for the document.
	ID string `json:"id,omitempty"`
	
	// CreatedAt is the time the document was created.
	CreatedAt time.Time `json:"created_at,omitempty"`
	
	// UpdatedAt is the time the document was last updated.
	UpdatedAt time.Time `json:"updated_at,omitempty"`
}

// NewDocument creates a new Document with the given content and metadata.
func NewDocument(content string, metadata map[string]interface{}) *Document {
	now := time.Now()
	return &Document{
		PageContent: content,
		Metadata:    metadata,
		CreatedAt:   now,
		UpdatedAt:   now,
	}
}

// WithID sets the ID of the document.
func (d *Document) WithID(id string) *Document {
	d.ID = id
	return d
}

// GetID returns the ID of the document, generating one if it doesn't exist.
func (d *Document) GetID() string {
	if d.ID == "" {
		// In a real implementation, generate a UUID
		d.ID = "doc_" + time.Now().Format("20060102150405")
	}
	return d.ID
}

// Update updates the document content and metadata.
func (d *Document) Update(content string, metadata map[string]interface{}) {
	d.PageContent = content
	
	if metadata != nil {
		if d.Metadata == nil {
			d.Metadata = metadata
		} else {
			// Merge metadata
			for k, v := range metadata {
				d.Metadata[k] = v
			}
		}
	}
	
	d.UpdatedAt = time.Now()
}

// Clone creates a deep copy of the document.
func (d *Document) Clone() *Document {
	clonedMetadata := make(map[string]interface{})
	for k, v := range d.Metadata {
		clonedMetadata[k] = v
	}
	
	return &Document{
		PageContent: d.PageContent,
		Metadata:    clonedMetadata,
		ID:          d.ID,
		CreatedAt:   d.CreatedAt,
		UpdatedAt:   d.UpdatedAt,
	}
}
